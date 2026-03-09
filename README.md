# Masked-Autoencoders-MAE-
Self-Supervised Image Representation Learning using Masked Autoencoders (MAE)


## Overview
This project implements a Masked Autoencoder (MAE) from scratch using base PyTorch layers for self-supervised visual representation learning. The model learns to reconstruct images with 75% of input patches masked, forcing it to develop deep semantic understanding without any labeled data.

 ## Architecture
The system follows an asymmetric transformer-based encoder-decoder design:
Encoder — ViT-Base (B/16)
ParameterValuePatch Size16 × 16Image Size224 × 224Hidden Dimension768Transformer Layers12Attention Heads12Parameters~86 Million

Accepts only 25% visible patches (49 out of 196)
Adds positional embeddings
Does NOT process mask tokens
Outputs latent representations for visible tokens only

Decoder — ViT-Small (S/16)
ParameterValueHidden Dimension384Transformer Layers12Attention Heads6Parameters~22 Million

Takes encoder latent tokens + learnable mask tokens
Reconstructs full patch sequence
Outputs pixel-level reconstruction


## Pipeline
Input Image (224×224)
       ↓
Split into 196 patches (16×16)
       ↓
Randomly mask 75% → 147 masked, 49 visible
       ↓
Encoder processes 49 visible patches
       ↓
Project to decoder dimension
       ↓
Append learnable mask tokens
       ↓
Decoder reconstructs all 196 patches
       ↓
MSE Loss on masked patches only

⚙️ Training Configuration
SettingValueDatasetTinyImageNet (200 classes)OptimizerAdamWLearning Rate1.5e-4Weight Decay0.05LR SchedulerCosine AnnealingBatch Size32Epochs50Mixed Precision✅ (torch.amp)Gradient Clipping1.0GPUsT4 × 2 (DataParallel)Mask Ratio0.75

 ## Project Structure
MAE/
├── genai_MAE.ipynb              # Main training notebook
├── mae-gradio-inference/
│   ├── app_gradio.py            # Gradio deployment app
│   └── README.md
├── recon_samples.png            # Qualitative reconstruction results
├── train_loss.png               # Training loss curve
└── metrics.txt                  # PSNR and SSIM scores

## Results
Qualitative Reconstruction
The model takes a 75% masked image and reconstructs the full image:
Masked InputReconstructionGround Truth75% patches removedModel outputOriginal image
See recon_samples.png for 5 sample reconstructions.
Quantitative Metrics
Evaluated on TinyImageNet validation set:
MetricScorePSNRSee metrics.txtSSIMSee metrics.txt

 ## Gradio App
An interactive demo app is included in mae-gradio-inference/:

Upload any image
Adjust masking ratio with a slider
View Masked Input, Reconstruction, and Ground Truth side by side

To run the app:
bashcd mae-gradio-inference
pip install gradio torch torchvision
python app_gradio.py

## Installation
bashpip install torch torchvision gradio matplotlib pillow
