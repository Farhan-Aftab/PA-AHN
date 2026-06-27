# ================================================================
# FILE 2: norms/prior_anchored_adaptive_hybrid_normalization.py
# Prior Anchored Adaptive Hybrid Normalization (PA-AHN)
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHybridNorm2d(nn.Module):
    """
    Adaptive Hybrid Normalization (AHN)
    - Parallel BN, LN, IN branches
    - Input-dependent controller using feature statistics [mean, variance]
    - Temperature-scaled softmax for adaptive fusion weights
    - Stores last weights for entropy-based stabilization and analysis

    Prior-Anchored Residual Controller upgrade:
    - Each layer learns a stable base normalization prior
    - Controller predicts only a residual correction
    - Final logits = base_logits + residual_scale * residual_logits
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        temperature: float = 0.7,
        safe_weights=(0.40, 0.33, 0.27),
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.temperature = temperature

        # ------------------------------------------------------------
        # Safe prior weights for early optimization stabilization
        # These are also used to initialize the learnable base logits
        # ------------------------------------------------------------
        safe_w = torch.tensor(safe_weights, dtype=torch.float32)
        safe_w = safe_w / safe_w.sum()
        self.register_buffer("safe_weights", safe_w)

        # Convert safe prior weights to logits for stable initialization
        init_base_logits = torch.log(safe_w + 1e-8)
        self.base_logits = nn.Parameter(init_base_logits.clone())

        # Learnable residual strength for this layer
        self.residual_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Normalization branches
        self.bn = nn.BatchNorm2d(num_features, affine=True)
        self.inorm = nn.InstanceNorm2d(num_features, affine=True)
        self.ln = nn.GroupNorm(1, num_features)  # LN-equivalent for NCHW tensors

        # Controller network (MLP): [mu, var] -> 3 logits
#        self.controller = nn.Sequential(
#            nn.Linear(2, 8),
#            nn.ReLU(),
#            nn.Linear(8, 3)
#        )                        #old

#        self.controller = nn.Sequential(
#            nn.Linear(2, 16),
#            nn.ReLU(),
#            nn.Linear(16, 16),
#            nn.ReLU(),
#            nn.Linear(16, 3)
#        )                        #new   But consider old before GAP implemented

        self.controller = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )                        #new after implementing GAP

        # For logging/visualization/entropy regularization
        self.last_weights = None
        self.last_entropy = None
        self.last_residual_logits = None
        self.last_base_weights = None

    def set_prior_weights(self, safe_weights: torch.Tensor) -> None:
        """
        Update the safe prior weights and reinitialize base logits accordingly.
        This lets training code enforce the same starting prior across layers.
        """
        safe_weights = safe_weights.detach().to(self.base_logits.device)
        safe_weights = safe_weights / safe_weights.sum()

        with torch.no_grad():
            self.safe_weights.copy_(safe_weights)
            self.base_logits.copy_(torch.log(safe_weights + 1e-8))

    def _compute_controller_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample controller input from raw convolution output.
        x shape: [B, C, H, W]
        returns: [B, 2] = [mean, variance]
        """
        # Mean and variance are computed from X_conv, not from normalized outputs
        mean = x.mean(dim=[2, 3]).mean(dim=1, keepdim=True)  # [B, 1]
        var = x.var(dim=[2, 3], unbiased=False).mean(dim=1, keepdim=True)  # [B, 1]
        controller_input = torch.cat([mean, var], dim=1)  # [B, 2]
        return controller_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Controller path
        # controller_input = self._compute_controller_input(x)         # [B, 2]
        # raw_scores = self.controller(controller_input)               # [B, 3]
        # weights = F.softmax(raw_scores / self.temperature, dim=1)    # [B, 3]  these 3 lines are old

        # controller_input = self._compute_controller_input(x)   old controller line before GAP

        pooled = torch.mean(x, dim=(2, 3))  # [B, C]
        controller_input = pooled   # change the controller_input  to GAP by these two lines

        # Residual controller output
        residual_logits = self.controller(controller_input)   # [B, 3]
        self.last_residual_logits = residual_logits

        # Base prior contribution
        base_logits = self.base_logits.unsqueeze(0).expand_as(residual_logits)

        # Prior-anchored residual controller:
        # final_logits = base_logits + residual_scale * residual_logits
        final_logits = base_logits + self.residual_scale * residual_logits

        weights = F.softmax(final_logits / self.temperature, dim=1)

        # Save prior-only weights for analysis
        with torch.no_grad():
            self.last_base_weights = F.softmax(self.base_logits, dim=0)

        # ADD THIS PART
        # weights = 0.85 * weights + 0.15 / 3.0
        # weights = weights / weights.sum(dim=1, keepdim=True)    # These 5 lines are new Change
        # Removed here because base prior already provides stabilization

        # Save for later use in entropy stabilization / plots
        self.last_weights = weights

        # Optional entropy tracking per layer
        eps = 1e-8
        entropy = -torch.sum(weights * torch.log(weights + eps), dim=1).mean()
        self.last_entropy = entropy

        # Parallel normalization branches
        out_bn = self.bn(x)
        out_ln = self.ln(x)
        out_in = self.inorm(x)

        # Per-sample adaptive fusion
        w_bn = weights[:, 0].view(-1, 1, 1, 1)
        w_ln = weights[:, 1].view(-1, 1, 1, 1)
        w_in = weights[:, 2].view(-1, 1, 1, 1)

        out = w_bn * out_bn + w_ln * out_ln + w_in * out_in
        return out

    def compute_entropy(self) -> torch.Tensor:
        """
        Returns last stored entropy for this AHN layer.
        Useful for entropy-based L-Stab in training.
        """
        if self.last_entropy is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.last_entropy
