import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import yaml
from pathlib import Path
import logging
from tqdm import tqdm

from ..model.durian import DurianModel
from ..utils.metrics import compute_metrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint

class DurianTrainer:
    """Trainer class for the Durian model."""
    
    def __init__(
        self,
        model: DurianModel,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the trainer.
        
        Args:
            model: Durian model instance
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"]
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(config["training"]["log_dir"])
        
        # Initialize logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                mel_target=batch["mel_target"],
                duration_target=batch["duration_target"]
            )
            
            # Compute loss
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["gradient_clip_val"]
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics = compute_metrics(outputs, batch)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                **{k: v.item() for k, v in metrics.items()}
            })
            
        # Compute average metrics
        avg_metrics = {
            "loss": total_loss / len(train_loader),
            **{k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0].keys()}
        }
        
        return avg_metrics
        
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    text=batch["text"],
                    text_lengths=batch["text_lengths"],
                    mel_target=batch["mel_target"],
                    duration_target=batch["duration_target"]
                )
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                
                # Update metrics
                total_loss += loss.item()
                metrics = compute_metrics(outputs, batch)
                
        # Compute average metrics
        avg_metrics = {
            "loss": total_loss / len(val_loader),
            **{k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0].keys()}
        }
        
        return avg_metrics
        
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the total loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Total loss
        """
        # Mel spectrogram loss
        mel_loss = F.mse_loss(outputs["mel_pred"], batch["mel_target"])
        
        # Duration loss
        if self.model.use_duration:
            duration_loss = F.mse_loss(
                outputs["duration_pred"],
                batch["duration_target"]
            )
        else:
            duration_loss = torch.tensor(0.0, device=self.device)
            
        # Total loss
        total_loss = mel_loss + duration_loss
        
        return total_loss
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ):
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train for
        """
        if num_epochs is None:
            num_epochs = self.config["training"]["max_epochs"]
            
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics["loss"])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.config["training"]["checkpoint_dir"],
                    "best"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config["training"]["early_stopping_patience"]:
                self.logger.info("Early stopping triggered")
                break
                
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
    ):
        """Log metrics to tensorboard and console.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch: Current epoch
        """
        # Log to tensorboard
        for name, value in train_metrics.items():
            self.writer.add_scalar(f"train/{name}", value, epoch)
            
        for name, value in val_metrics.items():
            self.writer.add_scalar(f"val/{name}", value, epoch)
            
        # Log to console
        self.logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        ) 