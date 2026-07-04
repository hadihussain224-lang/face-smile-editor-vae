# Smile Manipulation with a VAE (Variational Autoencoder)

Latent space editing on face images — making a face smile more or less
by moving its encoding along a learned "smile direction," without
retraining the model.

## Overview

Using a pretrained VAE trained on the CelebA dataset, this project computes
a smile attribute direction in latent space by comparing the average latent
codes of smiling vs. non-smiling faces. That direction vector is then added
to (or subtracted from) a new face's latent code to edit its expression —
a technique often called **latent space arithmetic**.

## How it works

1. Encode a face image into the VAE's latent space (mean vector `mu`)
2. Compute the smile direction: `mean(latent codes of smiling faces) − mean(latent codes of non-smiling faces)`
3. Add the direction vector, scaled by a strength factor `alpha`, to a face's latent code
4. Decode the edited latent code back into an image
5. Sweep `alpha` across a range (e.g. -4 to 4) to show the smile effect strengthening or reversing continuously

## Architecture

- VAE: Encoder → Sampler (reparameterization trick) → Decoder
- Latent dimension: 128
- Input resolution: 128×128 RGB

## Files

| File | Description |
|---|---|
| `Hadi_Hussain_CIS433_Project2.ipynb` | Full notebook: loading pretrained VAE, computing smile direction, latent edits, alpha sweep |
| `hadi_project2_model_submission.pt` | Trained/pretrained model weights |
| `Project2_Report.pdf` | Written report with results and analysis |

## Running it

```bash
pip install torch torchvision pandas pillow matplotlib
jupyter notebook Hadi_Hussain_CIS433_Project2.ipynb
```

## Dataset

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — large-scale
face attributes dataset (not included in this repo due to size; download
separately).

## Author

Muhammad Hadi Hussain
