import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ..model.durian import DurianModel
from ..config.config import Config
from .trainer import DurianTrainer
from ..data.dataset import DurianDataset
from ..utils.checkpoint import get_latest_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train Durian model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint to resume from"
    )
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration
    config = Config(args.config)
    
    # Create model
    model = DurianModel(**config.model_config)
    
    # Create datasets
    train_dataset = DurianDataset(
        data_dir=config.data_config["train_path"],
        **config.audio_config
    )
    val_dataset = DurianDataset(
        data_dir=config.data_config["val_path"],
        **config.audio_config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training_config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    trainer = DurianTrainer(
        model=model,
        config=config.config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Resume training if requested
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint or get_latest_checkpoint(
            config.training_config["checkpoint_dir"]
        )
        if checkpoint_path:
            trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed training from {checkpoint_path}")
        else:
            logger.warning("No checkpoint found, starting from scratch")
            
    # Train model
    trainer.train(train_loader, val_loader)
    
if __name__ == "__main__":
    main() 