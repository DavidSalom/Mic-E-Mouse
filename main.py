import torch
import sys
import wandb
from tqdm import tqdm
import numpy as np
import logging
import diffusers.schedulers


from core.VCTK_dataloader import setup_dataset
from core.WaveUNet import CrossAttentionWavUNet
from config import Config

# Parse arguments
cfg = Config(sys.argv[1:])

# Setup logging
logging_level = logging.ERROR if cfg.silent else  (logging.DEBUG if cfg.debug else logging.INFO) 
logging.basicConfig(level=logging_level, format='%(asctime)s %(levelname)s %(message)s')

# Setup dataset
train_loader, val_loader, test_loader = setup_dataset(cfg)

# Setup wandb
if cfg.wandb:
    wandb.init(
        project="MicEMouse",
        config = {
            "dataset": "VCTK",
            "resample_rate": cfg.resample_rate,
            "batch_size": cfg.batch_size,
            "train_split": cfg.train_split,
            "val_split": cfg.val_split,
            "test_split": cfg.test_split,
            "quantile": 0.95,
            "seed": cfg.seed,
            "N_diffusion": cfg.N_diffusion,
            "beta_min": cfg.beta_min,
            "beta_max": cfg.beta_max,
            "scheduler": cfg.scheduler,
            "beta_dist": "uniform",
            "learning_rate": cfg.learning_rate,
            "resumed": cfg.load_path is not None,
        },
    )
    
scheduler = diffusers.schedulers.__getattribute__(cfg.scheduler)(beta_start = cfg.beta_min, beta_end = cfg.beta_max, num_train_timesteps = cfg.N_diffusion)

# Setup model
net = CrossAttentionWavUNet().to(cfg.device)
if cfg.load_path is not None:
    net.load_state_dict(torch.load(cfg.load_path))
opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

for param in net.parameters():
    param.requires_grad = False

for param in net.out_res_conv.parameters():
    param.requires_grad = True

if cfg.validation:
    loader = val_loader
else:
    loader = train_loader
if not cfg.silent:
    pbar = tqdm(total=cfg.epochs)
epoch_losses = []
for epoch in range(cfg.epochs):
    losses = []
    windowed_losses = []
    i = 0
    N = len(loader)
    for x, _ in loader:
        x = x.to(cfg.device)
        b, _, _ = x.shape
        t = torch.randint(0, cfg.N_diffusion, (b,), device=cfg.device)
        eps = torch.randn_like(x, device=cfg.device)
        X = scheduler.add_noise(x, eps, t)
        eps_pred = net(X, t.unsqueeze(-1))
        loss = torch.mean((eps_pred - eps)**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        i += 1
        losses.append(loss.item())
    
        if not cfg.silent:
            pbar.set_description(f"Epoch {epoch + 1} Batch {i}/{N} loss: {loss.item():.4f}")
    
        if cfg.wandb:
            wandb.log({"loss": loss.item()})
            
    if cfg.wandb:
        wandb.log({"epoch_loss": np.mean(losses)})
        
    if not cfg.silent:
        pbar.update(1)
        
    epoch_losses.append(np.mean(losses))

# Save model
torch.save(net.state_dict(), f"{cfg.save_path}/model.pt")

if not cfg.silent:
    pbar.close()
if cfg.wandb:
    wandb.finish()
    