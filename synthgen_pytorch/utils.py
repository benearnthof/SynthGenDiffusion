"""Various Helper Functions"""
import torch
from torch import nn
from functools import reduce
from pathlib import Path

from imagen_pytorch.configs import ImagenConfig, ElucidatedImagenConfig
from ema_pytorch import EMA

def exists(val):
    return val is not None

def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)

def load_imagen_from_checkpoint(
    checkpoint_path,
    load_weights = True,
    load_ema_if_available = False
):
    model_path = Path(checkpoint_path)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'checkpoint not found at {full_model_path}'
    loaded = torch.load(str(model_path), map_location='cpu')

    imagen_params = safeget(loaded, 'imagen_params')
    imagen_type = safeget(loaded, 'imagen_type')

    if imagen_type == 'original':
        imagen_klass = ImagenConfig
    elif imagen_type == 'elucidated':
        imagen_klass = ElucidatedImagenConfig
    else:
        raise ValueError(f'unknown imagen type {imagen_type} - you need to instantiate your Imagen with configurations, using classes ImagenConfig or ElucidatedImagenConfig')

    assert exists(imagen_params) and exists(imagen_type), 'imagen type and configuration not saved in this checkpoint'

    imagen = imagen_klass(**imagen_params).create()

    if not load_weights:
        return imagen

    has_ema = 'ema' in loaded
    should_load_ema = has_ema and load_ema_if_available

    imagen.load_state_dict(loaded['model'])

    if not should_load_ema:
        print('loading non-EMA version of unets')
        return imagen

    ema_unets = nn.ModuleList([])
    for unet in imagen.unets:
        ema_unets.append(EMA(unet))

    ema_unets.load_state_dict(loaded['ema'])

    for unet, ema_unet in zip(imagen.unets, ema_unets):
        unet.load_state_dict(ema_unet.ema_model.state_dict())

    print('loaded EMA version of unets')
    return imagen

"""From https://github.com/atong01/conditional-flow-matching/"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons

# Implement some helper functions


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, noise=1.0, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.reverse = reverse
        self.noise = noise

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        if self.reverse:
            return -self.drift(x) + self.score(x)
        return self.drift(x) + self.score(x)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(t) * torch.ones_like(y) * self.noise
