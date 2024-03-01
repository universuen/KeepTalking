from __future__ import annotations

import torch

from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class OtherConfig:
    device: str = 'default'
    gemma_2b_lp_token: str = '<_lp>'

    def __post_init__(self) -> None:
        if self.device == 'default':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'  


@dataclass
class Gemma2bFullTrainingConfig:
    epochs: int = 1
    lr: float = 1e-2
    num_prompts: int = 50
    max_len: int = 50
    batch_size: int = 10


@dataclass
class Gemma2bPartialTrainingConfig:
    epochs: int = 1
    lr: float = 1e-2
    num_prompts: int = 100
    max_len: int = 100
    batch_size: int = 10


@dataclass
class PathConfig:
    project: Path = Path(__file__).absolute().parent
    src: Path = project / 'src'
    data: Path = project / 'data'
    scripts: Path = project / 'scripts'
    tests: Path = project / 'tests'
    logs: Path = data / 'logs'

    def __post_init__(self) -> None:
        for path in vars(self).values():
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggerConfig:
    level: int | str = logging.INFO
    path: Path = PathConfig().logs
