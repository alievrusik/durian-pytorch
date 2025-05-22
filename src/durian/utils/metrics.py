import torch
import torch.nn.functional as F
from typing import Dict, List

def compute_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute various metrics for model evaluation.
    
    Args:
        outputs: Model outputs
        batch: Input batch
        
    Returns:
        Dictionary containing computed metrics
    """
    metrics = {}
    
    # Mel spectrogram metrics
    metrics["mel_mse"] = F.mse_loss(outputs["mel_pred"], batch["mel_target"])
    metrics["mel_mae"] = F.l1_loss(outputs["mel_pred"], batch["mel_target"])
    
    # Duration metrics (if available)
    if "duration_pred" in outputs and "duration_target" in batch:
        metrics["duration_mse"] = F.mse_loss(
            outputs["duration_pred"],
            batch["duration_target"]
        )
        metrics["duration_mae"] = F.l1_loss(
            outputs["duration_pred"],
            batch["duration_target"]
        )
        
    return metrics

def compute_attention_metrics(
    attention_weights: torch.Tensor,
    text_lengths: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> Dict[str, float]:
    """Compute metrics for attention alignment.
    
    Args:
        attention_weights: Attention weights [batch_size, mel_length, text_length]
        text_lengths: Length of each text sequence [batch_size]
        mel_lengths: Length of each mel sequence [batch_size]
        
    Returns:
        Dictionary containing attention metrics
    """
    batch_size = attention_weights.size(0)
    metrics = {
        "attention_diagonality": 0.0,
        "attention_focus": 0.0
    }
    
    for i in range(batch_size):
        # Get valid attention weights
        attn = attention_weights[i, :mel_lengths[i], :text_lengths[i]]
        
        # Compute diagonality (how well aligned the attention is)
        diagonal = torch.diag(torch.ones(attn.size(0), device=attn.device))
        diagonality = torch.mean(torch.abs(attn - diagonal))
        metrics["attention_diagonality"] += diagonality.item()
        
        # Compute focus (how concentrated the attention is)
        focus = torch.mean(torch.max(attn, dim=1)[0])
        metrics["attention_focus"] += focus.item()
        
    # Average over batch
    metrics["attention_diagonality"] /= batch_size
    metrics["attention_focus"] /= batch_size
    
    return metrics 