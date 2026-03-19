import torch
import torch.nn as nn
import numpy as np

class MutantMLP(nn.Module):
    def __init__(self):
        super(MutantMLP, self).__init__()
        # Architecture: Input 72 (20 lidar + 4 physics * 3 frames), 64 ReLU, 64 ReLU, 2 Tanh
        self.net = nn.Sequential(
            nn.Linear(72, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

    def get_weights(self) -> np.ndarray:
        """
        Extracts all weights and biases from the network, flattens them,
        and concatenates them into a single 1D numpy array.
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, weights: np.ndarray):
        """
        Takes a 1D numpy array and perfectly maps the values back 
        into the exact shapes of the PyTorch network's parameter tensors.
        """
        idx = 0
        for param in self.parameters():
            shape = param.data.shape
            size = param.data.numel()
            param_weights = weights[idx:idx+size].reshape(shape)
            param.data = torch.tensor(param_weights, dtype=param.data.dtype, device=param.data.device)
            idx += size
