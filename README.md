# SEFH-LDCT: Low-Dose CT Image Denoising

## Title

Deep Learning-Based Denoising of Low-Dose CT Images Using Wasserstein Generative Adversarial Networks

## Abstract

Computed Tomography (CT) is one of the most widely used medical imaging modalities, providing critical diagnostic information for detecting cancers, cardiovascular diseases, and traumatic injuries. However, CT scans expose patients to ionizing radiation, which carries cumulative health risks including increased cancer probability. Low-Dose CT (LDCT) protocols reduce radiation exposure by up to 90%, but this reduction comes at the cost of increased image noise, which can compromise diagnostic accuracy. Deep learning offers a promising solution to this problem. By training neural networks to learn the mapping between noisy LDCT images and their high-quality Normal-Dose CT (NDCT) counterparts, we can computationally restore image quality without additional radiation exposure. This research is scientifically important as it advances the application of deep learning in medical imaging, and has significant societal impact by potentially enabling safer diagnostic imaging for millions of patients annually.

## Project Structure

```
SEF-LDCT/
├── train.py              # Main training script
├── evaluate.py           # Model evaluation script
├── preprocess_data.py    # Data preprocessing utilities
├── test_models.py        # Model testing
├── test_training.py      # Training pipeline tests
├── models/               # Model architectures
├── utils/                # Utility functions
│   ├── metrics.py        # Evaluation metrics
│   └── visualization.py  # Visualization tools
├── requirements.txt      # Python dependencies
└── dependencies.txt      # High-level dependency list
```

## Architecture

- Generator: U-Net architecture
  - Encoder: 3 encoder blocks (64, 128, 256 channels) with max pooling
  - Bottleneck: Convolutional block (512 channels)
  - Decoder: 3 decoder blocks (256, 128, 64 channels) with skip connections
  - Output: 1x1 conv with Tanh activation
  - Activation: LeakyReLU (0.2)
- Discriminator: PatchGAN Discriminator
- Loss Functions:
  - Wasserstein GAN loss with Gradient Penalty (WGAN-GP)
  - L1 reconstruction loss
  - Combined loss weights: λ_adv=1.0, λ_L1=100.0, λ_GP=10.0
    
## Training Configuration

- Optimizer: Adam (lr=0.0002, β1=0.5, β2=0.999)
- Batch size: 16
- Image size: 256×256
- n_critic: 5 (discriminator updates per generator update)
- Epochs: 150 (best results at epoch 120)
