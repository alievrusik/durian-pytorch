# Durian PyTorch

A PyTorch implementation of the Durian speech synthesis model. This project provides a professional, modular implementation of the Durian architecture for high-quality speech synthesis.

## Project Structure

```
durian-pytorch/
├── src/
│   └── durian/
│       ├── model/         # Model architecture components
│       ├── data/          # Data loading and preprocessing
│       ├── training/      # Training loops and utilities
│       ├── utils/         # Helper functions
│       └── config/        # Configuration management
├── tests/                 # Unit tests
└── configs/              # Experiment configurations
```

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python -m src.durian.training.train --config configs/default.yaml
```

### Inference

```bash
python -m src.durian.model.inference --checkpoint path/to/checkpoint.pt --text "Text to synthesize"
```

## Configuration

The project uses YAML configuration files to manage experiment settings. See `configs/default.yaml` for an example configuration.

## Development

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation when adding new functionality

## License

MIT License

## Citation

If you use this implementation in your research, please cite the original Durian paper:

```
@article{durian2020,
  title={Durian: Duration Informed Attention Network for Multimodal Synthesis},
  author={...},
  journal={...},
  year={2020}
}
```
