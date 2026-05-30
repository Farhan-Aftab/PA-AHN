# ================================================================
# FILE 2: norms/switchable_norm.py
# Switchable Normalization (SN)
# - Paper-faithful statistic-level mixing of IN/LN/BN
# - Separate learned weights for means and variances
# - Stores last weights for fair comparison/logging
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchableNorm2d(nn.Module):
    """
    Switchable Normalization (SN)

    Paper-faithful core logic:
    1. Compute IN, LN, BN statistics
    2. Learn separate importance weights for:
       - means   : [w_bn, w_ln, w_in]
       - variances: [w'_bn, w'_ln, w'_in]
    3. Mix statistics, not normalized branch outputs
    4. Apply a single affine transformation gamma, beta

    Notes:
    - This implementation keeps weights shared across channels,
      which matches the original paper design.
    - For logging compatibility with earlier code, last_weights stores
      the average of mean-weights and variance-weights.
    - Batch-average inference support is included through calibration.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Affine parameters (paper-style single affine after normalization)
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))   # gamma
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))    # beta

        # Learnable control logits for means and variances separately
        # Original paper uses separate weights for mean and variance.
        self.mean_weight_logits = nn.Parameter(torch.ones(3))  # [BN, LN, IN]
        self.var_weight_logits = nn.Parameter(torch.ones(3))   # [BN, LN, IN]

        # Running BN statistics for fallback inference
        self.register_buffer("running_mean_bn", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("running_var_bn", torch.ones(1, num_features, 1, 1))

        # Batch-average inference statistics (paper-style inference support)
        self.register_buffer("calib_mean_bn", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("calib_var_bn", torch.ones(1, num_features, 1, 1))
        self.register_buffer("calib_count", torch.tensor(0.0))

        self.use_calib_stats = False
        self.collect_batch_average = False

        # Temporary accumulators used during calibration
        self.register_buffer("calib_sum_mean_bn", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("calib_sum_var_bn", torch.zeros(1, num_features, 1, 1))

        # For logging / fair comparison with AHN
        self.last_weights = None
        self.last_mean_weights = None
        self.last_var_weights = None
        self.last_entropy = None

    def reset_batch_average(self) -> None:
        """
        Reset calibration accumulators before collecting batch-average stats.
        """
        self.calib_sum_mean_bn.zero_()
        self.calib_sum_var_bn.zero_()
        self.calib_count.zero_()
        self.use_calib_stats = False

    def start_batch_average_collection(self) -> None:
        """
        Enable collection of BN batch-average statistics.
        """
        self.collect_batch_average = True
        self.reset_batch_average()

    def stop_batch_average_collection(self) -> None:
        """
        Disable collection mode.
        """
        self.collect_batch_average = False

    def finalize_batch_average(self) -> None:
        """
        Finalize calibration statistics after collection.
        """
        if self.calib_count.item() > 0:
            self.calib_mean_bn.copy_(self.calib_sum_mean_bn / self.calib_count)
            self.calib_var_bn.copy_(self.calib_sum_var_bn / self.calib_count)
            self.use_calib_stats = True
        else:
            self.use_calib_stats = False

    def _compute_sn_statistics(self, x: torch.Tensor):
        """
        Compute IN, LN, BN statistics.
        Reuses IN statistics to compute LN and BN statistics efficiently.
        """
        # IN statistics: shape [N, C, 1, 1]
        mu_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = ((x - mu_in) ** 2).mean(dim=(2, 3), keepdim=True)

        # LN statistics based on IN statistics
        # mu_ln shape [N, 1, 1, 1]
        mu_ln = mu_in.mean(dim=1, keepdim=True)
        var_ln = (var_in + mu_in.pow(2)).mean(dim=1, keepdim=True) - mu_ln.pow(2)

        # ------------------------------------------------------------
        # BN statistics
        # During training: use current batch statistics
        # During inference: use calibrated batch-average stats if available,
        #                   otherwise use running stats
        # ------------------------------------------------------------
        if self.training:
            mu_bn = mu_in.mean(dim=0, keepdim=True)   # [1, C, 1, 1]
            var_bn = (var_in + mu_in.pow(2)).mean(dim=0, keepdim=True) - mu_bn.pow(2)

            # Update running stats for fallback inference
            self.running_mean_bn.mul_(1.0 - self.momentum).add_(self.momentum * mu_bn.detach())
            self.running_var_bn.mul_(1.0 - self.momentum).add_(self.momentum * var_bn.detach())

            # FIXED:
            # If calibration collection is enabled, accumulate while in training mode.
            if self.collect_batch_average:
                self.calib_sum_mean_bn.add_(mu_bn.detach())
                self.calib_sum_var_bn.add_(var_bn.detach())
                self.calib_count.add_(1.0)
        else:
            if self.use_calib_stats:
                mu_bn = self.calib_mean_bn
                var_bn = self.calib_var_bn
            else:
                mu_bn = self.running_mean_bn
                var_bn = self.running_var_bn

        return mu_in, var_in, mu_ln, var_ln, mu_bn, var_bn


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_in, var_in, mu_ln, var_ln, mu_bn, var_bn = self._compute_sn_statistics(x)

        # Separate weights for means and variances
        mean_weights = F.softmax(self.mean_weight_logits, dim=0)   # [3]
        var_weights = F.softmax(self.var_weight_logits, dim=0)     # [3]

        self.last_mean_weights = mean_weights.detach()
        self.last_var_weights = var_weights.detach()

        # For compatibility with previous logging scripts,
        # store the average of mean and variance weights.
        avg_weights = 0.5 * (mean_weights + var_weights)
        self.last_weights = avg_weights.detach().unsqueeze(0).repeat(x.size(0), 1)

        # Mixed mean and mixed variance
        # Weight order: [BN, LN, IN]
        mu = (
            mean_weights[0] * mu_bn +
            mean_weights[1] * mu_ln +
            mean_weights[2] * mu_in
        )

        var = (
            var_weights[0] * var_bn +
            var_weights[1] * var_ln +
            var_weights[2] * var_in
        )

        # Paper-style normalization
        out = (x - mu) / torch.sqrt(var + self.eps)
        out = out * self.weight + self.bias

        # Log entropy as average entropy of mean/var weights
        eps = 1e-8
        entropy_mean = -torch.sum(mean_weights * torch.log(mean_weights + eps))
        entropy_var = -torch.sum(var_weights * torch.log(var_weights + eps))
        self.last_entropy = 0.5 * (entropy_mean + entropy_var)

        return out


    def compute_entropy(self) -> torch.Tensor:
        if self.last_entropy is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.last_entropy
