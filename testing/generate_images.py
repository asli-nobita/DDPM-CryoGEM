import torch
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
from DDPM_model import DDPM
from train_ddpm import NoiseScheduler

def generate_images(model, scheduler, num_images, device, image_size=128):
    """
    Generate images using the trained DDPM model.
    
    Args:
        model: Trained DDPM model
        scheduler: Noise scheduler
        num_images: Number of images to generate
        device: Device to use for generation
        image_size: Size of the generated images
    """
    model.eval()
    images = []
    T = scheduler.num_timesteps

    with torch.no_grad():
        for _ in tqdm(range(num_images), desc="Generating images"):
            # starts with random noise
            x = torch.randn(1, 2, image_size, image_size).to(device)
            
            # reverse diffusion process
            for t in reversed(range(T)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                x = scheduler.p_sample(model, x, t_batch)
            
            # process the generated image
            img = x[0, 0].cpu().numpy()  # Take only the image channel
            img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1]
            img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Convert grayscale to RGB
            img_rgb = np.stack([img] * 3, axis=-1)  # Repeat grayscale channel 3 times
            images.append(Image.fromarray(img_rgb))
    
    return images

def main():
    parser = argparse.ArgumentParser(description='Generate images using trained DDPM model')
    parser.add_argument('--num_images', type=int, default=100,
                      help='Number of images to generate (default: 100)')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check available scheduling directories
    checkpoints_dir = "checkpoints"
    scheduling_dirs = [d for d in os.listdir(checkpoints_dir) if d.endswith('_scheduling')]
    
    if not scheduling_dirs:
        raise ValueError("No scheduling directories found in checkpoints folder")
    
    # Get the most recent scheduling directory
    scheduling_dir = max(scheduling_dirs, key=lambda d: os.path.getmtime(os.path.join(checkpoints_dir, d)))
    model_path = os.path.join(checkpoints_dir, scheduling_dir, "ddpm_best.pt")
    
    # Load checkpoint and get schedule type
    checkpoint = torch.load(model_path, map_location=device)
    schedule_type = checkpoint.get('schedule_type', 'cosine')  # Default to cosine if not found
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    print(f"Using noise schedule: {schedule_type}")
    
    # Initialize model
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type=schedule_type, device=device)
    
    generated_images = generate_images(model, scheduler, args.num_images, device)
    
    save_dir = os.path.join(checkpoints_dir, scheduling_dir, "generated_images")
    os.makedirs(save_dir, exist_ok=True)
    
    for i, img in enumerate(generated_images):
        img.save(os.path.join(save_dir, f"generated_{i}.png"))
    
    print(f"Generated {args.num_images} images and saved them to {save_dir}")

if __name__ == "__main__":
    main() 