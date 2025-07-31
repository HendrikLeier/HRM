"""
File for shared components between train and inference code
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from typing import Literal, Optional
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from torch.utils.data import DataLoader

import pydantic
from utils.functions import load_model_class

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig

DeviceSelector = Literal["cpu", "cuda"]

class ModelConfig(pydantic.BaseModel):
    """
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )
    """
    arch: ArchConfig
    batch_size: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    causal: bool
    device_type: DeviceSelector
    use_torch_attn: bool

def create_dataloader(config: PuzzleDatasetConfig, split: str):
    dataset = PuzzleDataset(config=config, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata

def create_model_wo_optimizers(model_config: ModelConfig, world_size: int) -> nn.Module:
    model_cls = load_model_class(model_config.arch.name)
    loss_head_cls = load_model_class(model_config.arch.loss.name)
    
    flat_nn_module_config = dict(
        **model_config.arch.__pydantic_extra__, # type: ignore
        **model_config.model_dump(exclude=set(["arch"]))
    )
    
    with torch.device(model_config.device_type):
        model: nn.Module = model_cls(flat_nn_module_config)
        model = loss_head_cls(model, **model_config.arch.loss.__pydantic_extra__)  # type: ignore
        
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore
        
        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
    
    return model
