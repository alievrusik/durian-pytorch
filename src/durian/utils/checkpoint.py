import torch
import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    name: str = "checkpoint",
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        metrics: Dictionary of metrics to save
        checkpoint_dir: Directory to save checkpoint
        name: Name of the checkpoint file
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    checkpoint_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
    }

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the path to the latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint if exists, None otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
        
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
        
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint) 