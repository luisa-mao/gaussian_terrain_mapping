import numpy as np
import torch

def get_covariance(scaling, yaw):
    n = scaling.shape[0]
    covariances = []

    for i in range(n):
        # Scaling matrix for each entry
        sx, sy = scaling[i]
        S = torch.tensor([[sx**2, 0], [0, sy**2]])

        # Rotation matrix for each yaw angle
        theta = yaw[i]
        R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                          [torch.sin(theta), torch.cos(theta)]])

        # Covariance matrix Σ = R * S * R.T
        R = R.float()
        S = S.float()
        covariance = R @ S @ R.T
        covariances.append(covariance)

    return torch.stack(covariances)