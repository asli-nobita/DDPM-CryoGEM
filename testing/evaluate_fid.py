import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from scipy import linalg
import os
from tqdm import tqdm
from DDPM_model import DDPM
from train_ddpm import NoiseScheduler

class InceptionV3FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def extract_features(self, images):
        features = []
        with torch.no_grad():
            for img in images:
                if isinstance(img, Image.Image):
                    img = self.transform(img)
                if img.dim() == 2:
                    img = img.unsqueeze(0)
                if img.size(0) == 1:
                    img = img.repeat(3, 1, 1)
                img = img.unsqueeze(0).to(self.device)
                feat = self.model(img)
                features.append(feat.cpu().numpy())
        return np.concatenate(features)

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean): covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def generate_images(model, scheduler, num_images, device, image_size=128):
    model.eval()
    images = []
    T = scheduler.num_timesteps

    with torch.no_grad():
        for _ in tqdm(range(num_images), desc="Generating images"):
            x = torch.randn(1, 2, image_size, image_size).to(device)
            for t in reversed(range(T)):
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                x = scheduler.p_sample(model, x, t_batch)
            img = x[0, 0].cpu().numpy()
            img = (img + 1) / 2
            img = (img * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(img))
    return images

def evaluate_fid(model_path, real_data_path, num_images=100, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    schedule_type = checkpoint.get('schedule_type', 'cosine')
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
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    print(f"Using noise schedule: {schedule_type}")

    scheduler = NoiseScheduler(num_timesteps=1000, schedule_type=schedule_type, device=device)

    feature_extractor = InceptionV3FeatureExtractor(device)

    real_images = []
    for img_name in os.listdir(real_data_path)[:num_images]:
        img_path = os.path.join(real_data_path, img_name)
        try:
            img = Image.open(img_path).convert('L')
            real_images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    if not real_images:
        raise ValueError("No valid real images found in the specified path")

    fake_images = generate_images(model, scheduler, num_images, device)

    real_features = feature_extractor.extract_features(real_images)
    fake_features = feature_extractor.extract_features(fake_images)

    fid = calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid:.2f}")

    save_dir = f"checkpoints/{schedule_type}_scheduling/generated_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(fake_images):
        img.save(os.path.join(save_dir, f"generated_{i}.png"))

    return fid

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule_type = "cosine"
    model_path = f"checkpoints/{schedule_type}_scheduling/ddpm_best.pt"
    real_data_path = "testing/data/Ribosome(10028)/real_data"
    fid = evaluate_fid(model_path, real_data_path, num_images=100, device=device)
