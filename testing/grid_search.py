import torch
import numpy as np
import os
from train_ddpm import train_ddpm, get_best_checkpoint
from evaluate_fid import evaluate_fid
from DDPM_model import DDPM
from dataset import get_data_loader
import pandas as pd
from tqdm import tqdm
import argparse

def run_experiment(num_epochs, learning_rate, iteration, device):
    """Run a single training experiment and return the FID score."""
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
    
    # Setup data loader
    batch_size = 32 if device == "cuda" else 16
    num_workers = 4 if device == "cuda" else 2
    dataloader = get_data_loader(
        "testing/data/Ribosome(10028)/real_data",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Create experiment directory
    exp_dir = f"experiments/epochs_{num_epochs}_lr_{learning_rate}_iter_{iteration}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Train model
    train_ddpm(
        model,
        dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=exp_dir
    )
    
    # Get best checkpoint
    best_checkpoint = get_best_checkpoint(exp_dir)
    
    # Calculate FID score
    fid = evaluate_fid(
        best_checkpoint,
        "testing/data/Ribosome(10028)/real_data",
        num_images=100,
        device=device
    )
    
    return fid

def main():
    parser = argparse.ArgumentParser(description='Grid search over epochs and learning rates')
    parser.add_argument('--num_iterations', type=int, default=5,
                      help='Number of iterations per configuration (default: 5)')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define grid search parameters
    epochs_list = [30, 50, 100]
    lr_list = [1e-4, 1.5e-4, 2e-4]
    
    # Initialize results dictionary
    results = {}
    
    # Run experiments
    for num_epochs in epochs_list:
        for lr in lr_list:
            key = f"{num_epochs}_{lr}"
            results[key] = []
            
            print(f"\nRunning experiments for {num_epochs} epochs, lr={lr}")
            for iteration in tqdm(range(args.num_iterations)):
                fid = run_experiment(num_epochs, lr, iteration, device)
                results[key].append(fid)
    
    # Create results table
    table_data = []
    for num_epochs in epochs_list:
        row = []
        for lr in lr_list:
            key = f"{num_epochs}_{lr}"
            fids = results[key]
            mean_fid = np.mean(fids)
            std_fid = np.std(fids)
            row.append(f"{mean_fid:.2f} ± {std_fid:.2f}")
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(
        table_data,
        index=[f"{e} epochs" for e in epochs_list],
        columns=[f"lr={lr}" for lr in lr_list]
    )
    
    # Save results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/grid_search_results.csv")
    print("\nResults saved to results/grid_search_results.csv")
    print("\nResults Table:")
    print(df)

if __name__ == "__main__":
    main() 