"""
Checking if given the same input values the pytorch and flash_attn implementation produce the same output
"""
from typing import Tuple

import math
import torch
import torch.nn.functional as func
from flash_attn import flash_attn_func

"""
# starting from [batch_size, seq_len, num_heads, head_dim]:
q = qkv_proj(hidden_states) ...  # yields [batch, seq, (num_heads+...), head_dim]
q = q.view(batch_size * num_heads, seq_len, head_dim)
k = k.view(       batch_size * num_key_value_heads, seq_len_kv, head_dim)
v = v.view(       batch_size * num_key_value_heads, seq_len_kv, head_dim)
"""

def get_random_qkv(
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # B * S * 3 * H * D
    fake_qkv = torch.rand((batch_size, seq_len, 3, head_dim, num_heads), dtype=torch.bfloat16).cuda()
    
    # reorder to [3, B, H, S, D]
    fake_qkv = fake_qkv.permute(2, 0, 3, 1, 4)
    
    # all shaped B * H * S * D
    q = fake_qkv[0]
    k = fake_qkv[1]
    v = fake_qkv[2]
    
    return q, k, v

def torch_flash_attn_transform(t: torch.Tensor) -> torch.Tensor:
    """
    Input format B * H * S * D
    Output format: B * S * H * D
    """
    return t.permute(0, 2, 1, 3)


def apply_functions(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Expecting q, k, v to be B * H * S * D
    """
    flash_tensor: torch.Tensor = flash_attn_func(q=torch_flash_attn_transform(q), k=torch_flash_attn_transform(k.clone()), v=torch_flash_attn_transform(v.clone())) # type: ignore
    flash_attn_result_in_torch_fmt = torch_flash_attn_transform(flash_tensor)
    
    torch_result = func.scaled_dot_product_attention(query=q, key=k, value=v)
    
    torch_result_cpu = func.scaled_dot_product_attention(query=q.cpu(), key=k.cpu(), value=v.cpu())

    print(torch.max(torch.abs(torch_result - flash_attn_result_in_torch_fmt)))
    print(torch.max(torch.abs(torch_result.cpu() - torch_result_cpu)))


if __name__ == "__main__":
    q, k, v = get_random_qkv(
        10, 10, 10, 10
    )
    
    apply_functions(q, k, v)
