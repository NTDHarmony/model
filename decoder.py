from typing import Optional, Tuple, Union, Callable
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache, EncoderDecoderCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from config import Config 
from norm import RMSNorm
from embedding import ChunkRotaryEmbedding, RotaryEmbedding, rotate_half, apply_rotary_pos_emb, apply_rotary_pos_emb_1
from attention import DualChunkAttention, CrossAttention
from feedforward import BlockSparseTop2MLP, SparseMoeBlock
from model.llm.glsa import GlobalLocalSparseAttention, create_fine_mask, create_sliding_mask
logger = logging.get_logger(__name__)


class DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        self.activation_fn = ACT2FN[config.hidden_act]
        self.attn = GlobalLocalSparseAttention(config, layer_idx).to("cuda").half()
        self.encoder_attn = CrossAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
            layer_idx=layer_idx,
            config=config,
        ).to("cuda").half()
        self.input_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to("cuda").half()
        self.encoder_attn_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to("cuda").half()
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim, bias=False)
        self.post_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to("cuda").half()
        self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to("cuda").half()

    def forward(
        self,
        hidden_states: torch.Tensor,
        sliding_window_flex_mask: Optional[torch.Tensor] = None,
        fine_selection_flex_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        position_embeddings: Optional[Tuple[torch.LongTensor, torch.LongTensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)

        result = self.attn(
            hidden_states = hidden_states,
            sliding_window_flex_mask=sliding_window_flex_mask,
            fine_selection_flex_mask=fine_selection_flex_mask,
            cache=cache,
            output_attentions=output_attentions,
            return_cache=use_cache,
        )
        if isinstance(result, tuple):
            hidden_states, attn_weights, present_key_value = result
        else:
            hidden_states = result
            attn_weights = None
            present_key_value = None

        hidden_states = nn.functional.dropout(hidden_states, p=0.1, training=self.training)
        hidden_states = residual + hidden_states

        # Cross Attention
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=0.1, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            if past_key_value is not None:
                present_key_value = past_key_value + cross_attn_present_key_value
            else:
                present_key_value = cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=0.1, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

class TransformerDecoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id
        self.layerdrop = config.layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.d_model = config.hidden_size
        self.num_codebooks = config.num_codebooks
        self.embed_dim = config.vocab_size + 1
        self.attn_sliding_window_size = config.sliding_window_size
        self.attn_fine_block_size = config.selection_block_size
        self.rotary_emb = RotaryEmbedding(
            config,
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            device='cuda',
        )
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(self.embed_dim, config.hidden_size) for _ in range (config.num_codebooks)]
        )
        self.rope = True
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        encoder_attn_implementation = config._attn_implementation
        encoder_attn_implementation = "eager"
        self.encoder_attn_implementation = encoder_attn_implementation
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
        
    def init_weights(self):
        initializer_factor = getattr(self.config, "initializer_range", 0.02)
        import torch.nn.init as init
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.trunc_normal_(module.weight, mean=0.0, std=initializer_factor,
                                a=-2 * initializer_factor, b=2 * initializer_factor)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0.0, std=initializer_factor)
                if hasattr(module, "padding_idx") and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

    
    def post_init(self):
        self.init_weights()
        #self._backward_compatibility_gradient_checkpointing()
        #PreTrainedModel._backward_compatibility_gradient_checkpointing(self)
        self._tp_plan = getattr(self.config, "base_model_tp_plan", None)
        self._pp_plan = getattr(self.config, "base_model_pp_plan", None)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        sliding_window_flex_mask: Optional[torch.Tensor] = None,
        fine_selection_flex_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = sum([self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks)])

        past_key_values_length = 0
        if cache_position is not None:
            past_key_values_length = cache_position[0]
        elif past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        input_shape = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                input_shape[1] + past_key_values_length,
                dtype = torch.long,
                device = inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)
            if position_ids.shape[1] > input_shape[1]:
                position_ids = position_ids[:, -input_shape[1]:]

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1]
            )

        """
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )
        """
        sliding_window_flex_mask = create_sliding_mask(input_shape[1], self.attn_sliding_window_size)
        fine_selection_flex_mask = create_fine_mask(input_shape[1], self.attn_fine_block_size)

        hidden_states = inputs_embeds
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = tuple(t.to(hidden_states.dtype) for t in position_embeddings)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = None
            if past_key_values is not None and len(past_key_values) > idx:
                past_key_value = past_key_values[idx]
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.forward,
                    hidden_states,
                    #causal_mask,
                    sliding_window_flex_mask,
                    fine_selection_flex_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    #position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    #attention_mask=causal_mask,
                    sliding_window_flex_mask=sliding_window_flex_mask,
                    fine_selection_flex_mask=fine_selection_flex_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    #position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache=cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if cross_attn_head_mask else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None


        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Config,
        past_key_values: Cache,
    ):

        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask