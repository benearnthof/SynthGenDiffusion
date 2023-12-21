"""
Vanilla Variational Autoencoder
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = "cuda"

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, eps=1e-3):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=eps)
        self.leakyrelu = nn.LeakyReLU(0.2) 
    def forward(self, x):
        return self.leakyrelu(self.batch_norm(self.conv(self.pad(self.upsample(x)))))

class VAE(nn.Module):
    """
    Vanilla Variational Autoencoder
    """
    def __init__(self, in_channels, encoder_shape, decoder_shape, latent_dim):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.encoder_shape = encoder_shape
        self.decoder_shape = decoder_shape
        self.latent_dim = latent_dim
        self.encoder_mults = [(1, 2), (2, 4), (4, 8), (8, 8)]
        self.encoder = nn.Sequential(ConvBlock(self.in_channels, self.encoder_shape))
        for m in self.encoder_mults:
            self.encoder.append(
                ConvBlock(self.encoder_shape * m[0], self.encoder_shape * m[1])
                )
        self.linear_mu = nn.Linear(self.encoder_shape*8*4*4, latent_dim)
        self.linear_logvar = nn.Linear(self.encoder_shape*8*4*4, latent_dim)
        self.decoder_mults = [(16, 8), (8, 4), (4, 2), (2, 1)]
        self.decoder = nn.Sequential()
        for m in self.decoder_mults:
            self.decoder.append(
                DecoderBlock(self.decoder_shape * m[0], self.decoder_shape * m[1])
            )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear_decode = nn.Linear(self.latent_dim, self.decoder_shape*8*2*4*4)
        self.out_conv = nn.Conv2d(self.decoder_shape, self.in_channels, 3, 1)
        self.out_pad = nn.ReplicationPad2d(1)
        self.out_upsample = nn.UpsamplingNearest2d(scale_factor=2)
    def encode(self, x):
        # convblocks
        for layer in self.encoder:
            x = layer(x)
        # linear transform
        x = x.view(-1, self.encoder_shape*8*4*4)
        return self.linear_mu(x), self.linear_logvar(x)
    def decode(self, z):
        # linear transform from latent shape
        h1 = self.relu(self.linear_decode(z))
        h1 = h1.view(-1, self.decoder_shape*8*2, 4, 4)
        for layer in self.decoder:
            h1 = layer(h1)
        # transform to normalized vector
        return self.sigmoid(self.out_conv(self.out_pad(self.out_upsample(h1))))
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device == "cuda":
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.encoder_shape, self.decoder_shape))
        z = self.reparametrize(mu, logvar)
        return z
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.encoder_shape, self.decoder_shape))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return z, res
