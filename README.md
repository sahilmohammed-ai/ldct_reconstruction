# SEF-LDCT: Low-Dose CT Image Denoising

Deep Learning-Based Denoising of Low-Dose CT Images Using Wasserstein Generative Adversarial Networks

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

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd SEF-LDCT
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

Due to file size limitations, the `data/` directory is not included in this repository.

**To use this project:**
- Place your LDCT dataset in the `data/` directory
- Run `preprocess_data.py` to prepare the data

## Training

Run training with:
```bash
python train.py
```

Model checkpoints will be saved to `checkpoints/` (not tracked in git).

## Evaluation

Evaluate trained models with:
```bash
python evaluate.py
```

Results will be saved to `results/` (not tracked in git).

## Configuration

Look at [BEST_CONFIGURATION.md](BEST_CONFIGURATION.md) for optimal training configurations.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

For full dependency list, see [requirements.txt](requirements.txt)
