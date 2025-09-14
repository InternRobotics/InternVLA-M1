import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, encoder_hidden_state, encoder_attention_mask=None):
        """
        query: [B, Q, D]
        encoder_hidden_state: [B, L, D]
        encoder_attention_mask: [B, L] or None
        """
        q = self.norm1(query)
        kv = encoder_hidden_state

        if encoder_attention_mask is not None:
            attn_mask = encoder_attention_mask.unsqueeze(1).to(dtype=torch.bool)  # [B, 1, L]
        else:
            attn_mask = None

        attn_output, _ = self.cross_attn(q, kv, kv, key_padding_mask=attn_mask)
        query = query + attn_output
        query = query + self.dropout(self.mlp(self.norm2(query)))
        return query


class LayerwiseQFormer(nn.Module):
    def __init__(self, input_hidden_dim=2048, output_hidden_dim=768, num_query_tokens=64, num_layers=37, num_heads=8, config=None):
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.output_hidden_dim = output_hidden_dim
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.config = config
        # 先把输入投影到 output_dim
        self.proj = nn.Linear(input_hidden_dim, output_hidden_dim)
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, output_hidden_dim))

        # 37 independent cross-attn blocks
        self.layers = nn.ModuleList([
            CrossAttentionBlock(output_hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, hidden_states_list, encoder_attention_mask=None):
        """
        hidden_states_list: list of encoder hidden states, each of shape [B, L, D]
        encoder_attention_mask: optional [B, L]
        return: updated query tokens [B, Q, D]
        """

        # hidden_states_list = self.scale_hook(hidden_states_list) #TODO 需要查看是否影响速度, 也需要check 是否影响性能

        assert len(hidden_states_list) == self.num_layers, f"Expected {self.num_layers} layers, got {len(hidden_states_list)}"

        B = hidden_states_list[0].size(0)
        # Project input hidden states to output dimension
        #    结果形状 [B, N, L, D_in]
        hs = torch.stack(hidden_states_list, dim=1)
        #    proj_hs 形状 [B, N, L, D_out]
        proj_hs = self.proj(hs)
        # 3) 拆回列表，每个元素恢复为 [B, L, D_out]
        hidden_states_list = list(proj_hs.unbind(dim=1))

        # Expand query tokens for each batch
        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        # Iterate through each layer and apply cross-attention
        for i, layer in enumerate(self.layers):
            query = layer(query, hidden_states_list[i], encoder_attention_mask)

        return query
    
    def scale_hook(self, hidden_states_list, scale_factor=0.1): #TODO 需要查看是否会影响分布式， 记得参数化
        # --- 1. 对输入 hidden_states_list 的梯度注册缩放钩子 ---
        # TODO @Jinhui 如果影响速度，需要用 lr 来曲线实现 --> 似乎是和 Deepspeed 加速冲突. lr 来曲线实现 这个不一定能够解，因为涉及到陡峭问题
        # TODO Jinhui： 即使要写， 也是写在 QFormer 里面
        if self.config and hasattr(self.config.vla, 'layer_qformer') and hasattr(self.config.vla.layer_qformer, 'grad_scale') \
        and self.config.vla.layer_qformer.grad_scale != 1 and False:
            scale_factor = self.config.vla.layer_qformer.grad_scale
        else:
            return hidden_states_list  # 如果没有配置 grad_scale，直接返回原始列表

        scaled_hidden_states_list = []
        for hidden_states in hidden_states_list:
            if hidden_states.requires_grad:
                # 确保在分布式环境下梯度缩放只执行一次
                if not hasattr(hidden_states, '_scaled_hook'):  # 防止重复注册 --> 看起来可以加速，
                    hook = lambda grad: grad * scale_factor
                    hidden_states.register_hook(hook)
                    hidden_states._scaled_hook = True  # 标记已处理
            scaled_hidden_states_list.append(hidden_states)

        return hidden_states_list


import torch
import torch.nn as nn

def get_layerwise_qformer(
    num_heads=8,
    config=None,
    **kwargs
):
    """
    Returns a LayerwiseQFormer model with specified parameters.

    """
    # dist.barrier()
    qformer_cfg = config.framework.layer_qformer
    num_layers = qformer_cfg.qformer_end_layer - qformer_cfg.qformer_start_layer  if config else num_layers
    num_query_tokens = qformer_cfg.num_query_tokens
    input_hidden_dim=config.framework.layer_qformer.input_dim
    output_hidden_dim=config.framework.layer_qformer.ouptput_dim
    num_query_tokens=qformer_cfg.num_query_tokens

    qformer = LayerwiseQFormer(input_hidden_dim=input_hidden_dim, output_hidden_dim=output_hidden_dim, num_query_tokens=num_query_tokens, num_layers=num_layers, num_heads=num_heads, config=config) 
    return qformer
