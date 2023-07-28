from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.optimizer import Optimizer
from typing import Optional

@dataclass
class TrainingConfig:
    n_epochs: int
    autoencoder_name: str
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    autoencoder: Optional[object] = None
    optimizer: Optional[Optimizer] = None
    use_openai: bool = False
    print_every: int = 150
    lr_scheduler: Optional[object] = None
    is_notebook: bool = True
    num_sentences: int = 1000
    save_path = None