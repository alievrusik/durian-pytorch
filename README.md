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

## External Dependencies

### GitHub Repositories

This project depends on two external repositories that are automatically installed via pip:

1. [Listener](https://github.com/shrubb/listener.git) - Used for audio processing
2. [WaveGlow](https://github.com/shrubb/waveglow.git) - Used for neural vocoding

Note: These repositories are forked versions of the original implementations. If you encounter any issues with these repositories, you may need to:
- Check for compatibility with your PyTorch version
- Ensure you have the correct CUDA version installed
- Consider using the original repositories if the forked versions don't work

### WaveGlow Model

The project requires a pre-trained WaveGlow model for inference. You can download it from NVIDIA's NGC:

```bash
# Download the WaveGlow model
wget https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/2/files/waveglow_256channels_ljs_v2.pt
```

Place the downloaded model file in the appropriate directory as specified in your configuration.

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
