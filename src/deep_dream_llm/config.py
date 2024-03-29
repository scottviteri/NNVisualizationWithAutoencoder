from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.optimizer import Optimizer
from typing import Optional

@dataclass
class TrainingConfig:
    autoencoder_name: str = None
    load_path: str = None
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    autoencoder: Optional[object] = None
    optimizer: Optional[Optimizer] = None
    use_openai: bool = False
    use_reencode: bool = False
    batch_size: int = 32
    latent_dim: float = 100
    learning_rate: float = 0.001
    lr_scheduler: Optional[object] = None
    is_notebook: bool = True
    num_sentences: int = 1000
    save_path: str = None
    lam : float = 0.5
