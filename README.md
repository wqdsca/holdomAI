# Poker Imitation Learning AI

An advanced poker AI system using imitation learning trained on the PHH (Poker Hand Histories) dataset.

## Features

- **Data Parser**: Robust PHH format parser for processing poker hand histories
- **Feature Extraction**: Comprehensive feature engineering for poker game states
- **Imitation Learning**: Deep learning model that learns from expert player decisions
- **Multiple Poker Variants**: Support for various poker games (Hold'em, Badugi, etc.)
- **Real-time Decision Making**: Fast inference for poker action selection

## Project Structure

```
poker_imitation_learning/
├── data/
│   ├── raw/           # Raw PHH dataset files
│   ├── processed/     # Processed and cached data
│   └── splits/        # Train/val/test splits
├── src/
│   ├── parser/        # PHH format parser
│   ├── features/      # Feature extraction
│   ├── models/        # Neural network models
│   ├── training/      # Training pipeline
│   └── evaluation/    # Testing and metrics
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks for analysis
└── scripts/           # Utility scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download PHH dataset
2. Parse and preprocess data: `python scripts/preprocess_data.py`
3. Train model: `python scripts/train.py`
4. Evaluate: `python scripts/evaluate.py`

## Model Architecture

The system uses a transformer-based architecture with:
- Position encoding for game sequence
- Multi-head attention for player interactions
- Action prediction head for decision making# holdomAI
