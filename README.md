## Objective

This project aims to generate a synthetic dataset by training a Denoising Diffusion Probabilistic Model (DDPM) on the [cryoGEM dataset](https://github.com/Cellverse/CryoGEM) containing grayscale images of Ribosome molecules, and then evaluating it using the Frechet Inception Distance metric (FID).

## 🔧 Dependencies and Installation

### Installation

1. Clone repo

```bash
git clone https://github.com/asli-nobita/DDPM-CryoGEM.git
```

2. (Optional) Create conda environment

```bash
conda create -n cryogem python=3.11
conda activate cryogem
```

3. Install all dependencies (make sure you are in the root directory)

```bash
pip install -e .
```

## Dataset

Install dataset from this link: [data.zip](https://www.dropbox.com/scl/fi/0zczm5hlb1h8qes1kobhz/data.zip?rlkey=46ob2ywa80t1mcvezy4lj6tu2&st=626po0mp&dl=0).  
Extract all files into `testing/data/`

## Working with the project

11. Open the Jupyter notebook and connect to a suitable kernel.

2. I used Google Colab when working, hence by default the drive is mounted. Change all directory paths accordingly. Here is the link to the Google Drive folder for reference: **[Drive folder](https://drive.google.com/drive/folders/1AKEMWG7X7bYcrA5K046MoQnngrtcltdr?usp=sharing)**

3. **All checkpoints were saved in `checkpoints/` directory, along with the comparison and loss curve visualizations. However due to storage limits in GitHub, checkpoints need to be downloaded from Google Drive directly at this link: [model_checkpoints](https://drive.google.com/drive/folders/14hZB_htaJDWn4IuffjRzWYNrA0kWdcFp?usp=sharing)**

   **Inside are two folders - one where I used linear beta scheduling for the noise generation, and the other where I used cosine beta scheduling. Each directory has its own set of checkpoints and generated images.**

4. The `testing/` directory contains the dataset (`data/`) as well as all the code for the project:

<!-- prettier-ignore-start -->
   - `DDPM_model.py` - describes the architecture of the DDPM model. This was sourced from [https://github.com/AAleka/retree](https://github.com/AAleka/retree). I did not alter their implementation.
   - `train_ddpm.py` - script for training the DDPM. This script also defines the noise scheduler for the DDPM. It creates a loss vs epoch curve, and saves the best checkpoint as well as saving checkpoints every 10 epochs. The hyperparameters `num_epochs` and `learning_rate` can be passed as command line arguments.
   - `generate_images.py` - script that generates a given number of synthetic images. This number is passed as a command line argument.
   - `evaluate_fid.py` - script that aims to evaluate FID between the real set of images and the generated dataset. It extracts features using the InceptionV3 model, loads real and generated images, and calculates FID using the formula  

     <p align="center">
     <img src="https://latex.codecogs.com/png.image?\dpi{110}\bg{white}$$FID=||\mu_1-\mu_2||^2&plus;\text{Tr}(\Sigma_1&plus;\Sigma_2-2(\Sigma_1\Sigma_2)^{1/2})$$" alt="FID Formula">
     </p>

     Afterwards, I tried taking a different approach by generating the images first and then calculating FID using the library function `pytorch_fid`.

   - `grid_search.py` - script that determines the ideal set of hyperparameters, `num_epochs` and `learning_rate` out of a given set of 9 values.
   - `visualize_comparison.py` - script that creates a side-by-side visualization of real and generated images, picking 10 random samples from both.
<!-- prettier-ignore-end -->
