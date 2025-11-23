# Social Relationship Recognition

Multimodal social relationship recognition using:
- Multi-Scale Feature Pyramid Networks
- Iterative Cross-Modal Refinement
- Uncertainty-Aware Adaptive Fusion

## Team Members
- [İsim 1] - [210201018]
- [İsim 2] - [ID]
- [İsim 3] - [ID]
- [İsim 4] - [ID]
- [İsim 5] - [ID]
- [İsim 6] - [ID]

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset
Download PISC dataset and place in `data/raw/`

## Project Structure
```
social_relationship_recognition/
├── data/           # Dataset and preprocessing
├── models/         # Model architectures
├── training/       # Training scripts
├── evaluation/     # Evaluation metrics
├── experiments/    # Experiment scripts
└── results/        # Experiment results
```

## Usage

### Train Baseline
```bash
python experiments/train_baseline.py
```

### Evaluate
```bash
python experiments/evaluate.py
```

## Progress
- [x] Environment setup
- [x] Literature review
- [ ] Baseline implementation
- [ ] Novel components
- [ ] Experiments
- [ ] Final report