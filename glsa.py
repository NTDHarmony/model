from typing import Optional, Callable, Tuple
from copy import deepcopy
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import arange, stack, cat, tensor, Tensor
from local_attention import LocalAttention #type: ignore
from rotary_embedding_torch import RotaryEmbedding #type: ignore
import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers.cache_utils import Cache, EncoderDecoderCache
from norm import RMSNorm
from config import Config

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

def create_sliding_mask(seq_len, window_size, causal=True):
    def sliding_mask(_, __, q_idx, kv_idx):
        distance = q_idx - kv_idx
        backward_sliding_mask = distance <= window_size
        forward_distance = 0 if causal else -window_size
        forward_sliding_mask = distance >= forward_distance
        causal_mask = backward_sliding_mask & forward_sliding_mask
        return causal_mask
    
    BLOCK_SIZE = int(window_size)
    Q_LEN = int(seq_len)
    KV_LEN = int(seq_len)
    block_mask = create_block_mask(
        sliding_mask, 
        B = None, H = None, 
        Q_LEN = Q_LEN, KV_LEN = KV_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
        _compile = True
    )
    return block_mask

def create_compress_mask(seq_len, kv_seq_len, compress_block_size, mem_kv_len = 0, causal = True):
    if not causal:
        return None

    def compress_mask(_, __, q_idx, kv_idx):
        is_mem_kv = kv_idx < mem_kv_len
        kv_without_mem = kv_idx - mem_kv_len
        compress_kv_idx = (kv_without_mem * compress_block_size) + (compress_block_size - 1)
        causal_mask = q_idx > compress_kv_idx
        return causal_mask | is_mem_kv

    BLOCK_SIZE = int(compress_block_size)
    block_mask = create_block_mask(
        compress_mask, 
        B = None, H = None, 
        Q_LEN = seq_len, 
        KV_LEN = kv_seq_len + mem_kv_len, 
        BLOCK_SIZE=BLOCK_SIZE,
        _compile = True
    )
    return block_mask

def create_fine_mask(seq_len, fine_block_size, causal = True):
    def inner(selected_block_indices: Tensor, num_grouped_queries = 1):
        device = selected_block_indices.device
        batch, kv_heads = selected_block_indices.shape[:2]
        one_hot_selected_block_indices = torch.zeros((*selected_block_indices.shape[:-1], seq_len // fine_block_size), device = device, dtype = torch.bool)
        one_hot_selected_block_indices.scatter_(-1, selected_block_indices, True)

        def fine_mask(b_idx, h_idx, q_idx, kv_idx):
            compressed_q_idx = q_idx // fine_block_size
            compressed_kv_idx = kv_idx // fine_block_size
            kv_head_idx = h_idx // num_grouped_queries
            is_selected = one_hot_selected_block_indices[b_idx, kv_head_idx, q_idx, compressed_kv_idx]
            if not causal:
                return is_selected
            causal_mask = q_idx >= kv_idx
            block_diagonal = compressed_q_idx == compressed_kv_idx
            return (causal_mask & (block_diagonal | is_selected))

        block_mask = create_block_mask(fine_mask, B = batch, H = kv_heads * num_grouped_queries, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
        return block_mask
    
    return inner

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def divisible_by(num, den):
    return (num % den) == 0

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)
    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def interpolate_1d(x, length, mode = 'bilinear'):
    x, inverse_pack = pack_one_with_inverse(x, '* n')
    x = rearrange(x, 'b n -> b 1 n 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    x = rearrange(x, 'b 1 n 1 -> b n')
    return inverse_pack(x)

def straight_through(t, target):
    return t + (target - t).detach()

def attend(q, k, v, mask = None, return_sim = False, scale = None):
    scale = default(scale, q.shape[-1] ** -0.5)
    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads
    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh = num_grouped_queries)
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    mask_value = max_neg_value(sim)
    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value)
    attn = sim.softmax(dim = -1)
    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')
    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')
    if not return_sim:
        return attn_out
    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')
    return attn_out, sim


class GlobalLocalSparseAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        assert config.num_key_value_heads <= config.num_attention_heads and divisible_by(config.num_attention_heads, config.num_key_value_heads)

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_grouped_queries = self.num_heads // self.num_kv_heads
        
        self.scale = self.head_dim ** -0.5

        dim = config.hidden_size
        dim_inner = self.head_dim * self.num_heads
        dim_kv_inner = self.head_dim * self.num_kv_heads
        self.norm = RMSNorm(dim)
        self.causal = config.causal
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.q_proj = nn.Linear(dim, dim_inner, bias=False)
        self.k_proj = nn.Linear(dim, dim_kv_inner, bias=False)
        self.v_proj = nn.Linear(dim, dim_kv_inner, bias=False)
        
        # Sliding Window Strategy
        self.sliding_window = LocalAttention(
            dim=self.head_dim,
            window_size=config.sliding_window_size,
            causal=True,
            exact_windowsize=True,
            autopad=True,
            use_rotary_pos_emb=False,
        )
        self.sliding_window_size = config.sliding_window_size
        
        # Compression Strategy
        self.compress_block_size = config.compress_block_size
        assert config.num_compressed_mem_kv > 0
        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n=self.compress_block_size)
        self.num_mem_compress_kv = config.num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, self.num_kv_heads, config.num_compressed_mem_kv, self.head_dim))
        self.k_intrablock_positions = nn.Parameter(torch.zeros(self.num_kv_heads, self.compress_block_size, self.head_dim))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(self.num_kv_heads, self.compress_block_size, self.head_dim))
        
        compress_mlp = config.compress_mlp
        if not exists(compress_mlp):
            compress_dim = self.compress_block_size * self.head_dim
            hidden_dim = int(config.compress_mlp_expand_factor * compress_dim)
            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.head_dim),
            )
        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)
        
        # Selection Strategy
        self.use_diff_topk = config.use_diff_topk
        self.interpolated_importance_score = config.interpolated_importance_score
        self.query_heads_share_selected_kv = config.query_heads_share_selected_kv
        self.selection_block_size = config.selection_block_size
        
        assert config.num_selected_blocks >= 0
        if config.num_selected_blocks == 0:
            print(f'`num_selected_blocks` should be set greater than 0, unless if you are ablating it for experimental purposes')
        
        self.num_selected_blocks = config.num_selected_blocks
        
        # Combination Strategy
        strategy_combine_mlp = config.strategy_combine_mlp
        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 3 * self.num_heads)
            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * self.num_heads))
        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h=self.num_heads)
        )

        # Head splitting and output projection
        self.split_heads = Rearrange('b n (h d) -> b h n d', d=self.head_dim)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim, bias=False)
        
    def _shape_query(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """Shape the query tensor to (bsz, num_heads, seq_len, head_dim)."""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_key_value(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """Shape the key/value tensor to (bsz, num_key_value_heads, seq_len, head_dim)."""
        return tensor.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]],
        return_cache = True,
    ):
        (
            (cache_key, cache_value), 
            (
                (cache_compress_key, cache_compress_value),
                (run_key, run_value)
            )
        ) = cache
        bsz, scale, heads, device = hidden_states.shape[0], self.scale, self.num_heads, hidden_states.device
        cache_len = cache_key.shape[-2]
        seq_len = cache_len + 1
        
        sliding_window = self.sliding_window_size
        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size
        
        hidden_states = self.norm(hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = self._shape_query(query_states, seq_len, bsz)  # (bsz, num_heads, seq_len, head_dim)
        key_states = self._shape_key_value(key_states, seq_len, bsz)  # (bsz, num_kv_heads, seq_len, head_dim)
        value_states = self._shape_key_value(value_states, seq_len, bsz) # (bsz, num_kv_heads, seq_len, head_dim)
        
        run_key = cat((run_key, key_states), dim=-2)
        run_value = cat((run_value, value_states), dim=-2)
        
        rotated_q = self.rotary_emb.rotate_queries_or_keys(query_states, offset = cache_len)
        key_states = self.rotary_emb.rotate_queries_or_keys(key_states, offset = cache_len)        
        
        key_states = cat((cache_key, key_states), dim=-2)
        value_states = cat((cache_value, value_states), dim=-2)
        
        if return_cache:
            cache_kv = (key_states, value_states)
        
        compressed_query = query_states
        compressed_key = cache_compress_key
        compressed_value = cache_compress_value
        
        repeated_compress_key = repeat(compressed_key, 'b h ... -> b (h gh) ...', gh=self.num_grouped_queries)
        repeated_compress_value = repeat(compressed_value, 'b h ... -> b (h gh) ...', gh=self.num_grouped_queries)
        
        compress_sim = einsum(query_states, repeated_compress_key, 'b h i d, b h j d -> b h i j') * scale
        compress_attn = compress_sim.softmax(dim=-1)
        
        compress_attn_out = einsum(compress_attn, repeated_compress_value, 'b h i j, b h j d -> b h i d')
        running_compress_seq_len = run_key.shape[-2]
        
        if divisible_by(seq_len, self.compress_block_size):
            key_compress_input = self.split_compress_window(key_states[..., -self.compress_block_size:, :] + self.k_intrablock_positions)
            value_compress_input = self.split_compress_window(value_states[..., -self.compress_block_size:, :] + self.v_intrablock_positions)
            
            next_compress_key = self.k_compress(key_compress_input)
            next_compress_value = self.v_compress(value_compress_input)
            
            run_key = run_key[..., 0:0, :]
            run_value = run_value[..., 0:0, :]
            
            compress_key = cat((compress_key, next_compress_key), dim=-2)
            compress_value = cat((compress_value, next_compress_value), dim=-2)
            
        if return_cache:
            cache_compress_kv = ((compress_key, compress_value), (run_key, run_value))
        
        assert self.compress_block_size == self.selection_block_size
        
        importance_scores = compress_sim[..., self.num_mem_compress_kv:]
        importance_scores += torch.randn_like(importance_scores) * 100
        
        num_compress_blocks = importance_scores.shape[-1]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0
        
        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(query_states, key_states)
        
        fine_sliding_window = (seq_len % self.selection_block_size) + 1
        fine_key = key_states[..., -fine_sliding_window:, :]
        fine_value = value_states[..., -fine_sliding_window:, :]
        
        if has_selected_kv_for_fine_attn:
            if self.query_heads_share_selected_kv:
                importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries=self.num_grouped_queries)
            slc_scores, slc_indices = importance_scores.topk(num_selected, dim=-1)
            
            fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
            remainder = fine_divisible_seq_len - key_states.shape[-2]
            
            slc_fine_key = pad_at_dim(rotated_k, (0, remainder), dim=-2)
            slc_fine_value = pad_at_dim(value_states, (0, remainder), dim=-2)
            
            slc_fine_key = rearrange(slc_fine_key, 'b h (w j) d -> b h w j d', j=self.selection_block_size)
            slc_fine_value = rearrange(slc_fine_value, 'b h (w j ) d -> b h w j d', j=self.selection_block_size)
            
            slc_indices = repeat(slc_indices, 'b h 1 slc -> b h slc j d', j=self.selection_block_size, d=slc_fine_key.shape[-1])
            
            slc_fine_key = slc_fine_key.gather(2, slc_indices)
            slc_fine_value = slc_fine_value.gather(2, slc_indices)
            
            slc_fine_key, slc_fine_value = tuple(rearrange(t, 'b h slc j d -> b h (slc j) d') for t in (slc_fine_key, slc_fine_value))
            
            fine_mask = slc_scores > 1e-10
            fine_mask = repeat(fine_mask, 'b h i slc -> b h i (slc j)', j=self.selection_block_size)
            
            fine_key = cat((slc_fine_key, fine_key), dim=-2)
            fine_value = cat((slc_fine_value, fine_value), dim=-2)
            fine_mask = F.pad(fine_mask, (0, fine_key.shape[-2] - fine_mask.shape[-1]), value=True)
        
        fine_query = rearrange(rotated_q, 'b (h gh) ... -> b h gh ...', gh=self.num_grouped_queries)
        fine_sim = einsum(fine_query, fine_key, 'b h gh i d, b h j d -> b h gh i j') * scale
        fine_sim = einx.where('b h i j, b h gh i j, -> b h gh i j', fine_mask, fine_sim, max_neg_value(fine_sim))
        fine_attn = fine_sim.softmax(dim=-1)
        
        fine_attn_out = einsum(fine_attn, fine_value, 'b h gh i j, b h j d -> b h gh i d')
        fine_attn_out = rearrange(fine_attn_out, 'b h gh ... -> b (h gh) ...')
        
        key_states = repeat(key_states, 'b h ... -> b (h gh) ...', gh=self.num_grouped_queries)
        value_states = repeat(value_states, 'b h ... -> b (h gh) ...', gh=self.num_grouped_queries)
        
        sliding_slice = (Ellipsis, slice(-(sliding_window + 1), None), slice(None))
        key_states, value_states  = key_states[sliding_slice], value_states[sliding_slice]
                
        sim = einsum(rotated_q, rotated_k, 'b h i d, b h j d -> b h i j') * scale
        attn = sim.softmax(dim=-1)
        sliding_window_attn_out = einsum(attn, value_states[sliding_slice], 'b h i j, b h j d -> b h i d')
        
        strategy_weighted_combine = self.to_strategy_combine(hidden_states)
        attn_out = einsum(strategy_weighted_combine, stack([compress_attn_out, fine_attn_out, sliding_window_attn_out]), 'b h n s, s b h n d -> b h n d')
        attn_out = self.merge_heads(attn_out)
        attn_out = self.combine_heads(attn_out)
        
        if not return_cache:
            return attn_out
        
        past_key_value = (cache_kv, cache_compress_kv)
        
        return attn_out, past_key_value
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        sliding_window_flex_mask: Optional[torch.Tensor] = None,
        fine_selection_flex_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        return_cache: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]]]:
        is_inferencing = exists(cache)
        if is_inferencing: 
            assert hidden_states.shape[1] == 1, 'input must be single tokens if inferencing with cache key values'
            return self.forward_inference(hidden_states, cache, return_cache=return_cache)
        
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        attn_weights = None
        
        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size
        
        # Convert input tensor to same dtype as model parameters
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        hidden_states = self.norm(hidden_states)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = self._shape_query(query_states, seq_len, bsz)  # (bsz, num_heads, seq_len, head_dim)
        key_states = self._shape_key_value(key_states, seq_len, bsz)  # (bsz, num_kv_heads, seq_len, head_dim)
        value_states = self._shape_key_value(value_states, seq_len, bsz) # (bsz, num_kv_heads, seq_len, head_dim)
        
    
        # compressed key / values - variables
        key_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)        
        value_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)        
        
        key_compress_input = self.split_compress_window(key_states[..., :compress_divisible_seq_len, :] + key_pos)
        value_compress_input = self.split_compress_window(value_states[..., :compress_divisible_seq_len, :] + value_pos)
        
        run_key = key_states[..., compress_divisible_seq_len:, :]
        run_value = value_states[..., compress_divisible_seq_len:, :]        
        
        compressed_query = query_states
        compressed_key = self.k_compress(key_compress_input)
        compressed_value = self.v_compress(value_compress_input)
        
        if return_cache: 
            cache_compress_kv = ((compressed_key, compressed_value), (run_key, run_value))  
        
        # 1. Coarse Attention over compressed
        mem_compressed_key, mem_compressed_value = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = bsz)
        num_mem_compress_kv = mem_compressed_key.shape[-2]
        
        compressed_key = cat((mem_compressed_key, compressed_key), dim=-2)
        compressed_value = cat((mem_compressed_value, compressed_value), dim=-2)
        compress_mask = None
        
        
        if self.causal:
            compressed_query_seq = arange(seq_len, device=device)
            compressed_key_seq = ((arange(num_compress_blocks, device=device) + 1) * self.compress_block_size) - 1
            compressed_key_seq = F.pad(compressed_key_seq, (num_mem_compress_kv, 0), value=-1)
            compress_mask = einx.less('j, i -> i j', compressed_key_seq, compressed_query_seq)
            
        compress_attn_out, compress_sim = attend(compressed_query, compressed_key, compressed_value, mask=compress_mask, return_sim=True)
        
        query_states, key_states = self.rotary_emb.rotate_queries_with_cached_keys(query_states, key_states)
        if return_cache:
            cache_kv = (key_states, value_states)
        
        # 2. fine attention over selected based on compressed attention logits
        importance_scores = compress_sim[..., num_mem_compress_kv:]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0
        
        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries=self.num_grouped_queries)
            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1
        
        if has_selected_kv_for_fine_attn:
            if self.compress_block_size != self.selection_block_size:
                compress_seq_len = num_compress_blocks * self.compress_block_size
                if self.interpolated_importance_score:
                    importance_scores = interpolate_1d(importance_scores, compress_seq_len)
                else:
                    importance_scores = repeat(importance_scores, '... j -> ... (j block_size)', block_size = self.compress_block_size)
                padding = fine_divisible_seq_len - compress_seq_len
                fine_query_seq_len = importance_scores.shape[-2]
                fine_query_padding = fine_divisible_seq_len - importance_scores.shape[-2]
                importance_scores = F.pad(importance_scores, (0, padding))
                
                block_causal_mask = torch.ones((num_fine_blocks,) * 2, device=device, dtype=torch.bool).tril(-1)
                block_causal_mask = repeat(block_causal_mask, 'i j -> (i n1) (j n2)', n1=self.selection_block_size, n2=self.selection_block_size)
                block_causal_mask = block_causal_mask[:fine_query_seq_len]
                
                importance_scores = importance_scores.masked_fill(~block_causal_mask, max_neg_value(compress_sim))
                importance_scores = reduce(importance_scores, '... (j block_size) -> ... j', 'mean', block_size=self.selection_block_size)
            
            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]
        
        fine_query = query_states
        fine_key = key_states
        fine_value = value_states
        
        if has_selected_kv_for_fine_attn:
            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim=-1)
            gates = None
            if self.use_diff_topk:
                gates = straight_through(selected_importance_values, 1.)
                #gates = gates.cumprod(dim=-1)[..., -1]
                #gates = repeat(gates, 'b h ... -> b (h qh) ...', qh=fine_num_grouped_queries)
            
            if exists(fine_selection_flex_mask):
                assert not self.use_diff_topk, 'differential topk is not available for flex attention'
                fine_block_mask = fine_selection_flex_mask(selected_block_indices, num_grouped_queries=fine_num_grouped_queries)
                fine_attn_out = flex_attention(fine_query, fine_key, fine_value, block_mask=fine_block_mask, enable_gqa=True)
            
            else:
                fine_mask = selected_importance_values > 1e-10
                if seq_len < fine_divisible_seq_len:
                    remainder = fine_divisible_seq_len - seq_len
                    fine_key = pad_at_dim(fine_key, (0, remainder), value=0., dim=-2)
                    fine_value = pad_at_dim(fine_value (0, remainder), value=0., dim=-2)
                    fine_query = pad_at_dim(fine_query, (0, remainder), value=0., dim=-2)
                    
                    fine_mask = pad_at_dim(fine_mask, (0, remainder), value=False, dim=-2)
                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value=0, dim=-2)
                    
                    if exists(gates):
                        gates = pad_at_dim(gates, (0, remainder), value=0, dim=-2)
                
                if self.causal:        
                    fine_window_seq = arange(fine_divisible_seq_len, device = device) // self.selection_block_size
                    fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b = bsz, h = selected_block_indices.shape[1])
                    selected_block_indices = cat((selected_block_indices, fine_window_seq), dim = -1) # for the block causal diagonal in fig2

                    fine_mask = repeat(fine_mask, 'b h i w -> b h i w j', j = self.selection_block_size)

                    causal_mask = torch.ones((self.selection_block_size,) * 2, device = device, dtype = torch.bool).tril()
                    causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w = num_fine_blocks, b = bsz, h = fine_mask.shape[1])

                    fine_mask = cat((fine_mask, causal_mask), dim = -2)
                    fine_mask = rearrange(fine_mask, 'b h i w j -> b h i (w j)')
                
                else: 
                    fine_mask = repeat(fine_mask, 'b h i w -> b h 1 i (w j)', j = self.selection_block_size)

                fine_key = rearrange(fine_key, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
                fine_value = rearrange(fine_value, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

                if self.query_heads_share_selected_kv:
                    fine_key = repeat(fine_key, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                    fine_value = repeat(fine_value, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                else:
                    fine_key = repeat(fine_key, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                    fine_value = repeat(fine_value, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fine_key.shape[-2], d = fine_key.shape[-1])

                fine_key = fine_key.gather(3, selected_block_indices)
                fine_value = fine_value.gather(3, selected_block_indices)
                
                if self.use_diff_topk:
                    if self.causal:
                        gates = F.pad(gates, (0, 1), value=1.)
                    fine_key = einx.multiply('b h i slc, b h i slc j d -> b h i slc j d', gates, fine_key)
                
                fine_key, fine_value = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fine_key, fine_value))
                fine_query = rearrange(fine_query, 'b (h qh) ... -> b h qh ...', qh = fine_num_grouped_queries)
                fine_sim = einsum(fine_query, fine_key, 'b h qh i d, b h i j d -> b h qh i j') * self.scale
                mask_value = max_neg_value(fine_sim)
                fine_sim = fine_sim.masked_fill(~fine_mask, mask_value)

                fine_attn = fine_sim.softmax(dim=-1)
                fine_attn_out = einsum(fine_attn, fine_value, 'b h qh i j, b h i j d -> b h qh i d')
                fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')
                fine_attn_out = fine_attn_out[..., :seq_len, :]
        
        else:
            seq_len = fine_key.shape[-2]
            fine_mask = None
            if self.causal:
                fine_mask = causal_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
            fine_attn_out = attend(fine_query, fine_key, fine_value, mask=fine_mask)
        
        # 3. overlapping sliding window
        sliding_query = query_states
        sliding_key = key_states
        sliding_value = value_states
        
        if exists(sliding_window_flex_mask):
            sliding_window_attn_out = flex_attention(sliding_query, sliding_key, sliding_value, block_mask=sliding_window_flex_mask, enable_gqa=True)
        else:
            sliding_key, sliding_value = tuple(
                repeat(t, 'b h ... -> b (h num_grouped_queries) ...', num_grouped_queries=self.num_grouped_queries) 
                for t in (sliding_key, sliding_value)
            )
            sliding_window_attn_out = self.sliding_window(sliding_query, sliding_key, sliding_value)
        
        # 4. combine strategies
        strategy_weighted_combine = self.to_strategy_combine(hidden_states)        
        attn_output = einsum(strategy_weighted_combine, stack([compress_attn_out, fine_attn_out, sliding_window_attn_out]), 'b h n s, s b h n d -> b h n d')
        attn_output = self.merge_heads(attn_output)
        attn_output = self.combine_heads(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        if not return_cache:
            return attn_output
        
        past_key_value = (cache_kv, cache_compress_kv)
            
        return attn_output, attn_weights, past_key_value