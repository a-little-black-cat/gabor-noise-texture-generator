"""Compatibility wrapper for the texture GAN generator."""

from model import TextureGAN


class Generator:
    def __init__(self, patch_size=64, latent_dim=128):
        self.model = TextureGAN(patch_size=patch_size, latent_dim=latent_dim).generator