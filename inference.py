"""
Because the training pipeline is using stuff that depends on nvcc (e.g. AdamTan2) it is not running on CPU only devices at all.
Training on CPU does not make sense, but it would be nice to be able to use CPU only to showcase some inference on trained models.
Thus this file implements a hardware agnostic inference method. It is based on the evaluate method but much leaner.
"""
import os

from typing import Optional, Dict, Tuple, Any
from shared import ModelConfig, ArchConfig, DeviceSelector, create_model_wo_optimizers
import torch.nn as nn
import torch
import pydantic
from omegaconf import DictConfig
import hydra


class CheckpointInfo(pydantic.BaseModel):
    checkpoint_path: str
    checkpoint_step: int


def init_model_for_inference(
    model_config: ModelConfig, checkpoint: Optional[CheckpointInfo]
) -> nn.Module:
    model = create_model_wo_optimizers(model_config=model_config, world_size=1)
    
    if checkpoint:
        checkpoint_file = os.path.join(checkpoint.checkpoint_path, f"step_{checkpoint.checkpoint_step}")
        checkpoint_obj = torch.load(checkpoint_file, map_location=model_config.device_type)
        model.load_state_dict(checkpoint_obj)
    
    return model


class InferenceResults(pydantic.BaseModel):
    metrics: Dict[str, Any]
    predictions: Dict[str, Any]


def infer(model_config: ModelConfig, model: nn.Module, input_data: Dict) -> InferenceResults:
    with torch.inference_mode():
        
        with torch.device(model_config.device_type):
            carry = model.initial_carry(batch) # type: ignore
            
        while True:
            carry, _, metrics, preds, all_finish = model(
                carry=carry, batch=input_data, return_keys=[
                    "logits", "q_halt_logits", "q_continue_logits", "target_q_continue"
                    ]
                )
            
            if all_finish:
                break
        
        return InferenceResults(
            metrics=metrics,
            predictions=preds
        )

class InferenceConfig(pydantic.BaseModel):
    arch: ArchConfig
    global_batch_size: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    causal: bool = False
    device_type: DeviceSelector = "cpu"
    checkpoint_path: Optional[str] = None

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
        
    inference_config = InferenceConfig(**hydra_config) # type: ignore
    
    model_cfg = ModelConfig(
        arch=inference_config.arch,
        batch_size=inference_config.global_batch_size,
        vocab_size=inference_config.vocab_size,
        seq_len=inference_config.seq_len,
        num_puzzle_identifiers=inference_config.num_puzzle_identifiers,
        causal=inference_config.causal,
        device_type=inference_config.device_type
    ) # type: ignore
    
    model = init_model_for_inference(model_cfg, checkpoint=None)

if __name__ == "__main__":
    launch()
