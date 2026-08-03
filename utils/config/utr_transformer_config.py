from dataclasses import dataclass,field
from typing import Optional, Tuple


@dataclass
class TransformerConfig:
    embed_dim: int = 768 
    num_attention_heads: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_type: str = "standard"
    performer_attention_features: int = 256
    num_layers: int = 12  
    max_seqlen: int = 1024

@dataclass
class OptimizerConfig:
      pass

@dataclass
class DataConfig:
    pass

@dataclass
class LoggingConfig:
    pass


@dataclass
class ProduceConfig:
    pass


@dataclass
class TrainConfig:
    pass



@dataclass
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: TransformerConfig = TransformerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    logging: LoggingConfig = LoggingConfig()
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False