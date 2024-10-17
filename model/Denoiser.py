import torch.nn as nn
from model.SinusoidalPosEmb import SinusoidalPosEmb
from model.ConvBlock import ConvBlock


class Denoiser(nn.Module):
    def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim=256, n_times=1000):
        super(Denoiser, self).__init__()
        _, _, img_C = image_resolution
        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)
        self.time_project = nn.Sequential(
            ConvBlock(diffusion_time_embedding_dim, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1))
        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])
        for idx in range(1, len(hidden_dims)):
            self.convs.append(
                ConvBlock(hidden_dims[idx - 1], hidden_dims[idx], kernel_size=3, dilation=3 ** ((idx - 1) // 2),
                          activation_fn=True, gn=True, gn_groups=8))
        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)

    def forward(self, perturbed_x, diffusion_timestep):
        y = perturbed_x
        diffusion_embedding = self.time_embedding(diffusion_timestep)
        diffusion_embedding = self.time_project(diffusion_embedding.unsqueeze(-1).unsqueeze(-2))
        y = self.in_project(y)
        for i in range(len(self.convs)):
            y = self.convs[i](y, diffusion_embedding, residual=True)
        y = self.out_project(y)
        return y