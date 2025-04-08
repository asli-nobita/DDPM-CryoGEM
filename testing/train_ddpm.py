import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from DDPM_model import DDPM
from dataset import get_data_loader
import glob
import matplotlib.pyplot as plt
import argparse

def plot_loss_curve(losses, save_dir):
    """Plot and save the loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

def get_best_checkpoint(checkpoint_dir):
    """Get the best checkpoint file from the directory based on loss value."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ddpm_best.pt"))
    if not checkpoint_files:
        return None
    return checkpoint_files[0]

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None, schedule_type='cosine'):
        """
        Initialize the noise scheduler with either linear or cosine beta schedule.
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting value for noise schedule (used for linear schedule)
            beta_end: Ending value for noise schedule (used for linear schedule)
            device: Device to store tensors on
            schedule_type: Type of noise schedule ('linear' or 'cosine')
        """
        self.num_timesteps = num_timesteps
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            # Define linear beta schedule
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        elif schedule_type == 'cosine':
            # Define cosine beta schedule
            t = torch.linspace(0, 1, num_timesteps + 1, device=self.device)
            f_t = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_bar = f_t / f_t[0]
            self.betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Calculate alphas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        
        # Calculate diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to image at timestep t
        Args:
            x_0: Original image
            t: Timestep
            noise: Predefined noise (if None, random noise will be generated)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def p_sample(self, model, x_t, t):
        """
        Reverse diffusion process: sample x_{t-1} from x_t
        Args:
            model: DDPM model
            x_t: Image at timestep t
            t: Current timestep
        """
        with torch.no_grad():
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # parameters for reverse process
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # mean for reverse distribution
            x_0_predicted = (x_t - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            x_0_predicted = torch.clamp(x_0_predicted, -1, 1)
            
            mean = (
                (x_t * (1 - beta) / torch.sqrt(alpha)) +
                (beta * x_0_predicted / torch.sqrt(alpha))
            )
            
            # adding variance
            if t > 0:
                noise = torch.randn_like(x_t)
                variance = torch.sqrt(beta) * noise
                x_t_prev = mean + variance
            else:
                x_t_prev = mean

            return x_t_prev

def get_save_dir(schedule_type):
    """Get the appropriate save directory based on schedule type."""
    return f"checkpoints/{schedule_type}_scheduling"

def train_ddpm(
    model,
    dataloader,
    num_epochs=30,
    learning_rate=2e-4,
    device=None,
    schedule_type='cosine',
    start_epoch=0,
    loss_history=None,
    best_loss=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    scheduler = NoiseScheduler(device=device, schedule_type=schedule_type)
    
    # Get save directory based on schedule type
    save_dir = get_save_dir(schedule_type)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training on device: {device}")
    print(f"Using {schedule_type} noise schedule")
    print(f"Starting from epoch: {start_epoch}")
    
    if loss_history is None:
        loss_history = []
    
    # Initialize best_loss from parameter or infinity if not provided
    if best_loss is None:
        best_loss = float('inf')
    print(f"Initial best loss: {best_loss:.4f}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device).long()
            
            # the forward diffusion process
            x_t, noise = scheduler.q_sample(batch[:, :1], t)  # Only add noise to image channel
            
            # Create input tensor with proper channels
            model_input = torch.cat([x_t, batch[:, 1:]], dim=1)
            
            predicted_noise = model(model_input, t)
            
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        plot_loss_curve(loss_history, save_dir)
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"ddpm_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'best_loss': best_loss,
                'schedule_type': schedule_type
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save best model if current loss is better
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(save_dir, "ddpm_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'best_loss': best_loss,
                'schedule_type': schedule_type
            }, checkpoint_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DDPM model')
    parser.add_argument('--num_epochs', type=int, default=30,
                      help='Number of training epochs (default: 30)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate (default: 2e-4)')
    parser.add_argument('--schedule_type', type=str, default='cosine',
                      choices=['linear', 'cosine'],
                      help='Type of noise schedule (default: cosine)')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training for {args.num_epochs} epochs with learning rate {args.learning_rate}")
    
    model = DDPM(
        time_dim=256,
        img_channels=2,  
        dim=2 * 2048,
        depth=1,
        heads=4,
        dim_head=64,
        mlp_ratio=4,
        drop_rate=0.
    ).to(device)
    
    batch_size = 32 if device == "cuda" else 16
    num_workers = 4 if device == "cuda" else 2
    
    dataloader = get_data_loader(
        "testing/data/Ribosome(10028)/real_data",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Check for existing best checkpoint
    save_dir = get_save_dir(args.schedule_type)
    best_checkpoint = get_best_checkpoint(save_dir)
    start_epoch = 0
    loss_history = None
    best_loss = None
    
    if best_checkpoint:
        print(f"Loading best checkpoint from: {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint.get('loss_history', [])
        best_loss = checkpoint.get('best_loss', None)
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Loaded loss history with {len(loss_history)} epochs")
        print(f"Best loss so far: {best_loss:.4f if best_loss is not None else 'N/A'}")
    
    train_ddpm(
        model, 
        dataloader, 
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        schedule_type=args.schedule_type,
        start_epoch=start_epoch, 
        loss_history=loss_history,
        best_loss=best_loss
    ) 