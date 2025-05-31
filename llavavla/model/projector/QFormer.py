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
    def __init__(self, input_hidden_dim=2048, output_hidden_dim=768, num_query_tokens=64, num_layers=37, num_heads=8):
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.output_hidden_dim = output_hidden_dim
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers

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
        hidden_states_list: list of 37 encoder hidden states, each of shape [B, L, D]
        encoder_attention_mask: optional [B, L]
        return: updated query tokens [B, Q, D]
        """
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
    


import torch
import torch.nn as nn

class QFormer(nn.Module):
    def __init__(self, hidden_dim=2048, num_query_tokens=64, num_layers=6, num_heads=8):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_dim = hidden_dim

        # learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim))

        # Transformer layers with cross-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, encoder_hidden_states):
        """
        encoder_hidden_states: [B, L, D] from QwenVL (qwenvl_feature.hidden_states[0])
        returns: updated query tokens [B, num_queries, D]
        """
        B = encoder_hidden_states.size(0)
        # Expand query tokens for each batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, D]

        # Cross-attention:
        # In PyTorch's TransformerEncoder, it only supports self-attention.
        # So we concatenate query+encoder, and mask out self-attention on encoder tokens if needed.

        # Here we simply concatenate queries and encoder tokens as input
        input_cat = torch.cat([queries, encoder_hidden_states], dim=1)  # [B, num_queries + L, D]

        # Run through transformer
        output = self.transformer(input_cat)  # [B, num_queries + L, D]

        # Slice back updated query tokens
        updated_queries = output[:, :self.num_query_tokens, :]  # [B, num_queries, D]
        return updated_queries
    

import torch
import torch.nn as nn

class ActionFeatureProjector(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, T, D] → [B, D', T]
        """
        x = self.proj(x)           # [B, T, out_dim]
        x = x.permute(0, 2, 1)     # → [B, out_dim, T]
        return x



def get_qformer(
    hidden_dim=2048,
    num_query_tokens=64,
    num_layers=6,
    num_heads=8
):
    """
    Returns a QFormer model with specified parameters.
    """
    return QFormer(
        hidden_dim=hidden_dim,
        num_query_tokens=num_query_tokens,
        num_layers=num_layers,
        num_heads=num_heads
    )
def get_layerwise_qformer(
    input_hidden_dim=2048,
    output_hidden_dim=768, #TODO 这里要和action model 对齐的
    num_query_tokens=64,
    num_layers=6,
    num_heads=8
    # 你应该全程允许参数config 进来 @Jinhui TODO 
):
    """
    Returns a LayerwiseQFormer model with specified parameters.

    """
    # dist.barrier()
    qformer = LayerwiseQFormer(input_hidden_dim=input_hidden_dim, output_hidden_dim=output_hidden_dim, num_query_tokens=num_query_tokens, num_layers=num_layers, num_heads=num_heads) 
    return qformer