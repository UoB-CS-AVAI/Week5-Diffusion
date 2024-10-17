import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def show_image(x, idx):
    fig = plt.figure()
    image_data = x[idx].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
    image_data = np.clip(image_data, 0, 1)
    plt.imshow(image_data)


def draw_sample_image(x, postfix):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Visualization of {postfix}")
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))


def visualise_forward_process_by_t(diffusion, x, num_timesteps_to_show):
    x = diffusion.scale_to_minus_one_to_one(x)
    timesteps = torch.linspace(0, diffusion.n_times - 1, steps=num_timesteps_to_show).long().to(diffusion.device)
    fig, axs = plt.subplots(1, num_timesteps_to_show, figsize=(20, 4))
    for idx, t in enumerate(timesteps):
        t = t.unsqueeze(0)
        perturbed_images, _ = diffusion.make_noisy(x, t)
        axs[idx].imshow(diffusion.reverse_scale_to_zero_to_one(perturbed_images)[0, 0].cpu().numpy())
        axs[idx].axis('off')
        axs[idx].set_title(f't={t.item()}')
    plt.show()
