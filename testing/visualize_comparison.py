import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def load_real_images(data_dir, num_images=10):
    """Load random real images from the dataset."""
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    images = []
    for file in selected_files:
        img_path = os.path.join(data_dir, file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = transforms.ToTensor()(img)
        images.append(img)
    
    return torch.stack(images)

def load_generated_images(schedule_type='cosine', num_images=10):
    """Load random generated images from the generated_images directory."""
    image_dir = f"checkpoints/{schedule_type}_scheduling/generated_images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    images = []
    for file in selected_files:
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path).convert('L')
        img = transforms.ToTensor()(img)
        images.append(img)
    
    return torch.stack(images)

def denormalize_image(img):
    """Convert normalized tensor to PIL Image."""
    img = img.squeeze().cpu().numpy()
    img = (img + 1) / 2  # Scale from [-1, 1] to [0, 1]
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def visualize_comparison(real_images, generated_images, schedule_type='cosine', save_path=None):
    """Create a side-by-side visualization of real and generated images."""
    if save_path is None:
        save_path = f"checkpoints/{schedule_type}_scheduling/comparison.png"
    
    num_images = len(real_images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5*num_images))
    
    for i in range(num_images):
        # Plot real image
        real_img = denormalize_image(real_images[i])
        axes[i, 0].imshow(real_img, cmap='gray')
        axes[i, 0].set_title('Real Image')
        axes[i, 0].axis('off')
        
        # Plot generated image
        gen_img = denormalize_image(generated_images[i])
        axes[i, 1].imshow(gen_img, cmap='gray')
        axes[i, 1].set_title('Generated Image')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison visualization saved to {save_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get schedule type from checkpoint
    schedule_type = 'cosine'  # Default to cosine if not specified
    checkpoint_path = f"checkpoints/{schedule_type}_scheduling/ddpm_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        schedule_type = checkpoint.get('schedule_type', 'cosine')
        print(f"Using {schedule_type} noise schedule")
    
    real_data_dir = "testing/data/Ribosome(10028)/real_data"
    real_images = load_real_images(real_data_dir, num_images=10)
    generated_images = load_generated_images(schedule_type, num_images=10)
    
    visualize_comparison(real_images, generated_images, schedule_type)

if __name__ == "__main__":
    main() 