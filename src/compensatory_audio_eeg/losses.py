import numpy as np
import torch
from torch import nn

STATE_LABELS = ["CS", "MS", "AF"]

class StateBalancedLoss(nn.Module):

    def __init__(self, y_state, y_target, n_target, condition_weight_power=0.5, label_smoothing=0.0):
        super().__init__()
        y_state = np.asarray(y_state)
        y_target = np.asarray(y_target)
        state_counts = np.bincount(y_state, minlength=len(STATE_LABELS)).astype(np.float32)
        state_counts[state_counts == 0] = 1.0
        state_weights = state_counts.sum() / (len(STATE_LABELS) * state_counts)
        state_weights = np.power(state_weights, condition_weight_power)
        state_weights = state_weights / state_weights.mean()
        target_counts = np.bincount(y_target, minlength=n_target).astype(np.float32)
        target_counts[target_counts == 0] = 1.0
        target_weights = target_counts.sum() / (n_target * target_counts)
        self.register_buffer('state_weights', torch.tensor(state_weights, dtype=torch.float32))
        self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32))
        self.label_smoothing = label_smoothing

    def forward(self, logits, target, state):
        ce = nn.functional.cross_entropy(logits, target, weight=self.target_weights, label_smoothing=self.label_smoothing, reduction='none')
        sample_weights = self.state_weights[state]
        return (ce * sample_weights).sum() / sample_weights.sum().clamp_min(1e-06)
