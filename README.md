# Doodle-to-Real-Image-Translation-and-Colorization-using-Pix2Pix
# Pix2Pix — Doodle-to-Real Image Translation & Colorization

**Course:** Generative AI (AI4009) | **Assignment:** No. 3 — Question 2 | **Semester:** Spring 2026  
**Authors:** Bushra Abad (22F-3324) & Tayyaba Imtiyaz (22F-3863)  
**Institution:** National University of Computer and Emerging Sciences (NUCES)

---

## Table of Contents

1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Training Configuration](#training-configuration)
8. [Results & Evaluation](#results--evaluation)
9. [Gradio App](#gradio-app)
10. [Key Findings](#key-findings)

---

## Overview

This project implements **Pix2Pix**, a Conditional Generative Adversarial Network (cGAN) for paired image-to-image translation. Two translation tasks are tackled:

- **CUHK Task:** Face Sketch → Realistic Photo
- **Anime Task:** Anime Sketch → Colored Image

Pix2Pix learns a supervised mapping from input images to output images using paired training data, combining an adversarial loss with an L1 reconstruction loss to produce sharp, structurally faithful outputs.

---

## Datasets

| Dataset | Task | Source |
|---|---|---|
| CUHK Face Sketch Database (CUFS) | Sketch → Realistic Face Photo | [Kaggle](https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs) |
| Anime Sketch Colorization Pair | Anime Sketch → Colored Image | [Kaggle](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) |

**Preprocessing applied to both datasets:**
- Resized all images to **256 × 256**
- Normalized pixel values to **[-1, 1]**
- Applied random horizontal flipping for data augmentation
- Loaded via custom PyTorch `Dataset` and `DataLoader` classes

---

## Model Architecture

### Generator — U-Net

The generator follows a U-Net encoder-decoder design with skip connections.

- **Input:** Sketch / Edge / Grayscale image (3 channels, 256×256)
- **Output:** Realistic photo or colored image (3 channels, 256×256)
- **Encoder:** Successive downsampling convolutional blocks with LeakyReLU
- **Decoder:** Successive upsampling transposed convolutional blocks with ReLU + Dropout
- **Skip connections:** Each encoder layer is concatenated with its corresponding decoder layer to preserve fine spatial details
- **Output activation:** Tanh

### Discriminator — PatchGAN

The discriminator classifies overlapping **16×16 patches** of the image as real or fake rather than scoring the full image.

- **Input:** Concatenated (sketch, target) pair — either real or generated
- **Architecture:** Series of convolutional layers with LeakyReLU and BatchNorm
- **Output:** Matrix of probabilities (one per patch)
- **Advantage:** Enforces local texture realism and penalizes high-frequency artifacts

---

## Project Structure

```
22f_3324_22f_3863_AI_ASS03_Pix2Pix/
│
├── 22f_3324_22f_3863_AI_ASS03_Pix2Pix_v3.ipynb   # Main Jupyter Notebook
├── README.md                                        # This file
│
├── checkpoints/
│   ├── gen_cuhk_epoch_*.pth                         # CUHK generator checkpoints
│   └── gen_anime_epoch_*.pth                        # Anime generator checkpoints
│
└── outputs/
    ├── training_curves_cuhk.png                     # Loss plots — CUHK
    ├── training_curves_anime.png                    # Loss plots — Anime
    ├── samples_cuhk.png                             # Visual results — CUHK
    └── samples_anime.png                            # Visual results — Anime
```

---

## Setup & Installation

### Platform
Trained on **Kaggle** with **GPU T4 x2** accelerator.

### Dependencies

```bash
pip install torch torchvision gradio scikit-image kagglehub
```

All other dependencies (NumPy, Matplotlib, PIL) come pre-installed in the Kaggle/Colab environment.

### Dataset Download (via kagglehub inside notebook)

```python
import kagglehub
cuhk_path  = kagglehub.dataset_download("arbazkhan971/cuhk-face-sketch-database-cufs")
anime_path = kagglehub.dataset_download("ktaebum/anime-sketch-colorization-pair")
```

---

## How to Run

1. Open `22f_3324_22f_3863_AI_ASS03_Pix2Pix_v3.ipynb` in Kaggle or Google Colab.
2. Set the accelerator to **GPU T4**.
3. Run all cells in order:
   - **Section 1–2:** Install dependencies and set hyperparameters
   - **Section 3–4:** Download datasets and explore structure
   - **Section 5:** Define custom Dataset classes
   - **Section 6:** Build Generator (U-Net) and Discriminator (PatchGAN)
   - **Section 7:** Define loss functions (GAN + L1)
   - **Section 8–9:** Train on CUHK dataset
   - **Section 10–11:** Train on Anime dataset
   - **Section 12:** Visualize results
   - **Section 13:** Launch Gradio app

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Image Size | 256 × 256 |
| Batch Size | 16 |
| Epochs | 30 |
| Learning Rate | 0.0002 |
| Adam Betas | (0.5, 0.999) |
| L1 Loss Weight (λ) | 100 |
| Mixed Precision | ✅ (`torch.cuda.amp`) |
| Checkpoint Interval | Every 5 epochs |

### Loss Functions

**Total Generator Loss:**
```
L_total = L_adversarial + λ * L_L1
```

- `L_adversarial` — Standard GAN loss (BCE); encourages the generator to fool the discriminator
- `L_L1` — Pixel-wise L1 reconstruction loss; ensures outputs are structurally close to ground truth
- `λ = 100` — Weighs L1 heavily to prevent blurry but structurally wrong outputs

**Discriminator Loss:**
```
L_D = L_real + L_fake
```

---

## Results & Evaluation

### Quantitative Metrics

| Metric | CUHK (Sketch → Photo) | Anime (Sketch → Color) |
|---|---|---|
| SSIM ↑ | Computed per epoch | Computed per epoch |
| PSNR (dB) ↑ | Computed per epoch | Computed per epoch |

Metrics are computed using `scikit-image` on the validation split after each training run.

### Qualitative Results

The visualization module (`visualize()`) displays a 3-row grid for `n=5` samples:

```
Row 1: Input Sketch
Row 2: Generated Output  (model prediction)
Row 3: Ground Truth      (target image)
```

Both models are evaluated separately — CUHK for face photo synthesis and Anime for sketch colorization.

---

## Gradio App

A live interactive demo is built with **Gradio** inside the notebook.

**Features:**
- Upload any sketch or edge image
- Select translation mode:
  - `CUHK — Sketch → Photo` (face reconstruction)
  - `Anime — Sketch → Color` (anime colorization)
- View the generated output in real time

```python
demo = gr.Interface(
    fn=translate,
    inputs=[gr.Image(type='pil'), gr.Radio(choices=[...])],
    outputs=gr.Image(type='pil'),
    title='Pix2Pix — Sketch to Image Translation'
)
demo.launch()
```

The app automatically generates a public shareable URL when launched on Colab/Kaggle.

---

## Key Findings

- The **L1 loss** is critical to structural fidelity — without it, the generator produces visually plausible but spatially incorrect outputs.
- **PatchGAN** significantly improves local texture sharpness compared to a global discriminator.
- **Skip connections** in the U-Net preserve fine edge details (facial features, line art) that a plain encoder-decoder would lose.
- The Anime task converged faster than the CUHK task due to more consistent coloring patterns in anime art.
- Mixed precision (`torch.cuda.amp`) reduced memory usage by ~30%, allowing a batch size of 16 at 256×256 resolution on a single T4 GPU.

---

## References

- Isola, P., et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
- CUHK Face Sketch Database (CUFS) — [Kaggle](https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs)
- Anime Sketch Colorization Pair — [Kaggle](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair)

