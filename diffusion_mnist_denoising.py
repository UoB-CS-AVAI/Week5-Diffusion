from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model.Denoiser import Denoiser
from model.utils import show_image, draw_sample_image, visualise_forward_process_by_t

def gaussian_window(size, sigma):
    """Generate a 1D Gaussian window."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()  # Normalize to make the sum of all elements 1
    return g

def create_window(window_size, sigma, channel):
    """Generate a 2D Gaussian window and adapt it for multiple channels."""
    _1D_window = gaussian_window(window_size, sigma)
    _2D_window = _1D_window[:, None] * _1D_window[None, :]  # Outer product to create a 2D window
    _2D_window = _2D_window.view(1, 1, window_size, window_size)
    return _2D_window.expand(channel, 1, window_size, window_size)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1
        self.window = create_window(window_size, sigma, self.channel)

    def forward(self, img1, img2):
        # Ensure that the window has the same type as the input images (for mixed precision training)
        self.window = self.window.to(img1.dtype).to(img1.device)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2.pow(2)
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1 * mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map = torch.clamp(ssim_map, 0, 1)  # Clamp SSIM values to the range [0, 1]
        return ssim_map.mean()

    def ssim_loss(self, img1, img2):
        return 1 - self.forward(img1, img2)

class Diffusion(nn.Module):
    def __init__(self, model, image_resolution=[28, 28, 1], n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):
        super(Diffusion, self).__init__()
        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)
        # Define cosine variance schedule (betas)
        # betas = self.cosine_schedule(n_times).to(device)  # Use cosine schedule instead of linear
        self.sqrt_betas = torch.sqrt(betas)
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.device = device

    def cosine_schedule(self, num_timesteps, s=0.008):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2

        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.02)
        return betas

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros, t):
        epsilon = torch.randn_like(x_zeros).to(self.device)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        return noisy_sample.detach(), epsilon

    def forward(self, x_zeros):
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        B, _, _, _ = x_zeros.shape
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        pred_epsilon = self.model(perturbed_images, t)
        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, timestep, t):
        B, _, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)
        epsilon_pred = self.model(x_t, timestep)
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z
        return x_t_minus_1.clamp(-1., 1)

    def sample(self, N):
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)
        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        return x_0

def train(diffusion, train_loader, optimizer, epochs, device, loss_type='mse'):
    diffusion.train()
    scaler = torch.cuda.amp.GradScaler()

    if loss_type == 'mse':
        loss_func = nn.MSELoss()
    elif loss_type == 'mae':
        loss_func = nn.L1Loss()
    elif loss_type == 'ssim':
        loss_func = SSIMLoss().to(device)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    for epoch in range(epochs):
        noise_prediction_loss = 0
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", ncols=100) as pbar:
            for batch_idx, (x, _) in pbar:
                optimizer.zero_grad()
                x = x.to(device)
                with torch.cuda.amp.autocast():
                    noisy_input, epsilon, pred_epsilon = diffusion(x)
                    loss = loss_func(pred_epsilon, epsilon)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                noise_prediction_loss += loss.item()
                pbar.set_postfix({"Denoising Loss": f"{noise_prediction_loss / (batch_idx + 1):.6f}"})
        torch.cuda.empty_cache()
    print("Training finished!")

if __name__ == "__main__":
    # Set device
    print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if DEVICE.type == "cuda":
        torch.cuda.set_device(0)

    # Model Hyperparameters
    config = {
        'dataset': 'MNIST',
        'img_size': (28, 28, 1),
        'timestep_embedding_dim': 256,
        'n_layers': 8,
        'hidden_dim': 256,
        'n_timesteps': 1000,  # test
        'beta_minmax': [1e-4, 2e-2],  # test
        'train_batch_size': 128,
        'inference_batch_size': 64,
        'lr': 5e-5,
        'epochs': 1,  # test
        'seed': 42,
    }

    hidden_dims = [config['hidden_dim'] for _ in range(config['n_layers'])]
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_dataset = MNIST('./data/', transform=transform, train=True, download=True)
    test_dataset = MNIST('./data/', transform=transform, train=False, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['train_batch_size'], shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['inference_batch_size'], shuffle=False, **kwargs)

    # Initialize Model and optimizer
    model = Denoiser(image_resolution=config['img_size'],
                     hidden_dims=hidden_dims,
                     diffusion_time_embedding_dim=config['timestep_embedding_dim'],
                     n_times=config['n_timesteps']).to(DEVICE)

    diffusion = Diffusion(model, image_resolution=config['img_size'], n_times=config['n_timesteps'],
                          beta_minmax=config['beta_minmax'], device=DEVICE).to(DEVICE)

    optimizer = optim.Adam(diffusion.parameters(), lr=config['lr'])

    # Visualising forward process by timesteps
    model.eval()  # Set model to evaluation mode
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(DEVICE)
        perturbed_images, epsilon, pred_epsilon = diffusion(x)
        perturbed_images = diffusion.reverse_scale_to_zero_to_one(perturbed_images)
        visualise_forward_process_by_t(diffusion, x, num_timesteps_to_show=10)
        break
    show_image(perturbed_images, 0)
    plt.show()

    # Training
    print("Start training DDPMs for denoising...")
    train(diffusion, train_loader, optimizer, config['epochs'], DEVICE,
          loss_type='mae')  # loss type (e.g., 'mse', 'mae', 'ssim')

    # Evaluation and Sampling
    model.eval()
    with torch.no_grad():
        generated_images = diffusion.sample(N=config['inference_batch_size'])
    show_image(generated_images, idx=0)
    plt.show()

    # Visualisation
    draw_sample_image(perturbed_images, "Perturbed Images")
    draw_sample_image(generated_images, "Generated Denoised Images")
    draw_sample_image(next(iter(test_loader))[0][:config['inference_batch_size']], "Ground-truth Images")
    plt.show()