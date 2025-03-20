from typing import TYPE_CHECKING, Any, List, Dict, List, Optional, Tuple, Union, Callable
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.utils.checkpoint
import copy
import inspect
from dataclasses import dataclass
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache, EncoderDecoderCache
from transformers.generation import (
    ClassifierFreeGuidanceLogitsProcessor,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationMode,
    MinLengthLogitsProcessor
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutput,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList, MaxLengthCriteria 
from transformers.utils import LossKwargs, logging, ModelOutput
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForTextEncoding
from config import Config, ModelConfig
from decoder import DecoderLayer, TransformerDecoder
from logits_processor import CustomLogitsProcessor 
from xcodec import XCodecConfig, XCodecModel

if TYPE_CHECKING: 
    from transformers.generation.streamers import BaseStreamer
logger = logging.get_logger(__name__)
NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache, "sliding_window": SlidingWindowCache}


@dataclass
class CausalLMModelOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    codebook_losses: Optional[List[torch.FloatTensor]] = None
    
@dataclass
class Seq2SeqOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_states: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    codebook_losses: Optional[Tuple[List[torch.FloatTensor]]] = None

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class TransformerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.decoder = TransformerDecoder(config)
        self.post_init()
    
    def post_init(self):
        self._tp_plan = getattr(self.config, "base_model_tp_plan", None)
        self._pp_plan = getattr(self.config, "base_model_pp_plan", None)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


class CausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.generation_config = GenerationConfig(
            bos_token_id=config.bos_token_id if hasattr(config, 'bos_token_id') else 1025,
            eos_token_id=config.eos_token_id if hasattr(config, 'eos_token_id') else 1024,
            pad_token_id=config.pad_token_id if hasattr(config, 'pad_token_id') else 1024,
            decoder_start_token_id=config.decoder_start_token_id if hasattr(config, 'decoder_start_token_id') else 1025
        )
        self.model = TransformerModel(config)
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )
        self.post_init()
    
    def post_init(self):
        self._tp_plan = getattr(self.config, "base_model_tp_plan", None)
        self._pp_plan = getattr(self.config, "base_model_pp_plan", None)

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMModelOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (labels is not None) and (input_ids is None and inputs_embeds is None):
            input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.bos_token_id)
            
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        loss = None
        codebook_losses = None
        if labels is not None:
            logits = lm_logits[:, :, -labels.shape[1] :]
            loss_fct = CrossEntropyLoss()
            # Get device from input tensors
            device = input_ids.device if input_ids is not None else hidden_states.device
            loss = torch.zeros([], device=device)
            codebook_losses = []

            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)
            #mask = (input_ids.transpose(1,2) != self.config.eos_token_id) & ((labels != -100))

            for codebook in range(self.num_codebooks):
                codebook_logits = logits[:, codebook].contiguous().view(-1, logits.shape[-1])
                codebook_labels = labels[..., codebook].contiguous().view(-1)
                codebook_loss = loss_fct(codebook_logits, codebook_labels)
                codebook_losses.append(codebook_loss)
                loss += codebook_loss

            loss = loss / self.num_codebooks

        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMModelOutputWithCrossAttentions(
            loss=loss,
            last_hidden_states=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            codebook_losses=codebook_losses,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None, 
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        delay_pattern_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # apply the delay pattern mask
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        # Reshape input_ids from [batch, num_codebooks, seq_len] to [batch*num_codebooks, seq_len]
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            num_codebooks = input_ids.shape[1]
            seq_len = input_ids.shape[2]
            input_ids = input_ids.view(batch_size * num_codebooks, seq_len)
            
            # Also reshape/expand position_ids if provided
            if position_ids is not None:
                position_ids = position_ids.unsqueeze(1).expand(batch_size, num_codebooks, -1)
                position_ids = position_ids.reshape(batch_size * num_codebooks, -1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values, 
            "use_cache": use_cache,
            "inputs_embeds": inputs_embeds,
        }

    def build_delay_pattern_mask(self, input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int, max_length: int = None):
        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
        bsz, num_codebooks, seq_len = input_ids.shape

        max_length = max_length if max_length is not None else self.generation_config.max_length
        if max_length is None or max_length < seq_len:
            max_length = seq_len
        
        # For shifting, if audio_channels==1, we need room for shifting up to (num_codebooks - 1) positions.
        if self.config.audio_channels == 1:
            if max_length < seq_len + num_codebooks - 1:
                max_length = seq_len + num_codebooks - 1
        # For stereo (audio_channels==2), assume codebooks come in pairs.
        elif self.config.audio_channels == 2:
            channel_codebooks = num_codebooks // 2
            if max_length < seq_len + channel_codebooks - 1:
                max_length = seq_len + channel_codebooks - 1
                
        #input_ids_shifted = (torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1)
        input_ids_shifted = torch.full((bsz, num_codebooks, max_length), -1, dtype=torch.long, device=input_ids.device)

        channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
        if max_length < 2 * channel_codebooks - 1:
            return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

        for codebook in range(channel_codebooks):
            if self.config.audio_channels == 1:
                start = codebook
                end = seq_len + codebook
                input_ids_shifted[:, codebook, start:end] = input_ids[:, codebook]
            else:
                start = codebook
                end = seq_len + codebook
                input_ids_shifted[:, 2 * codebook, start:end] = input_ids[:, 2 * codebook]
                input_ids_shifted[:, 2 * codebook + 1, start:end] = input_ids[:, 2 * codebook + 1]

        eos_delay_pattern = torch.triu(torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=max_length - channel_codebooks + 1)
        bos_delay_pattern = torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))
        
        if self.config.audio_channels == 2:
            bos_delay_pattern = bos_delay_pattern.repeat_interleave(2, dim=0)
            eos_delay_pattern = eos_delay_pattern.repeat_interleave(2, dim=0)
            
        bos_mask = ~(bos_delay_pattern).to(input_ids.device)
        eos_mask = ~(eos_delay_pattern).to(input_ids.device)
        mask = ~(bos_delay_pattern | eos_delay_pattern).to(input_ids.device)
        #input_ids = mask * input_ids_shifted + ~bos_mask * bos_token_id + ~eos_mask * pad_token_id
        bos_token_id_int = bos_token_id[0] if isinstance(bos_token_id, (list, tuple)) else bos_token_id
        pad_token_id_int = pad_token_id[0] if isinstance(pad_token_id, (list, tuple)) else pad_token_id
        input_ids = mask * input_ids_shifted + (~bos_mask).to(input_ids_shifted.dtype) * bos_token_id_int + (~eos_mask).to(input_ids_shifted.dtype) * pad_token_id_int

        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            first_start_id = seq_len

        pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
        return input_ids, pattern_mask

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = input_ids.shape[0] // self.num_codebooks
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=input_ids.device)

        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        #model_kwargs["guidance_scale"] = generation_config.guidance_scale

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, generation_config, model_kwargs
            )

        # 5. Prepare max_length depending on other stopping criteria
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            logger.warning(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation."
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_length >= generation_config.max_length:
            logger.warning(
                f"Input length of decoder_input_ids is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=input_ids,
            input_ids_length=input_ids_length,
        )

        # 6. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=generation_config.max_length,
        )

        if streamer is not None:
            streamer.put(input_ids.cpu())

        model_kwargs["delay_pattern_mask"] = delay_pattern_mask

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        # 8. prepare batched CFG externally
        #if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
        #    logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
        #    generation_config.guidance_scale = None

        # 9. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids.device,
        )

        # 10. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                **model_kwargs,
            )

            # 11. run sample
            outputs = self._sample(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )


        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        output_ids = self.apply_delay_pattern_mask(output_ids, model_kwargs["delay_pattern_mask"])

        _, mask = self.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=output_ids.shape[1],
        )

        mask = (mask != generation_config.bos_token_id) & (mask != generation_config.pad_token_id)
        output_ids = output_ids[mask].reshape(batch_size, self.num_codebooks, -1)

        if generation_config.return_dict_in_generate:
            outputs.sequences = output_ids
            return outputs
        else:
            return output_ids
        
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]) -> None:
        """Validates model kwargs for generation. Checks if all the keys are expected."""
        model_kwargs_copy = model_kwargs.copy()
        allowed_types = (torch.Tensor, bool, int, str, float, list, tuple, dict)
        
        if "past_key_values" in model_kwargs_copy:
            model_kwargs_copy.pop("past_key_values")
        if "position_ids" in model_kwargs_copy:
            model_kwargs_copy.pop("position_ids")
        if "cache_position" in model_kwargs_copy:
            model_kwargs_copy.pop("cache_position")
        if "delay_pattern_mask" in model_kwargs_copy:
            model_kwargs_copy.pop("delay_pattern_mask")
            
        forward_params = inspect.signature(self.forward).parameters
        for k, v in model_kwargs_copy.items():
            if k not in forward_params and v is not None:
                if not isinstance(v, allowed_types):
                    raise ValueError(
                        f"Generating with `{k}` is not supported. Only the following types are supported:"
                        f" {allowed_types}"
                    )
                logger.warning(
                    f"Generating with `{k}`. This argument has no effect on generation when using this model and"
                    " will be removed in a future version."
                )

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, str, Dict[str, torch.Tensor]]:
        if model_kwargs is None:
            model_kwargs = {}
        inputs = self._maybe_initialize_input_ids_for_generation(
            inputs=inputs, bos_token_id=bos_token_id, model_kwargs=model_kwargs
        )
        return inputs, "input_ids", model_kwargs

    def _prepare_special_tokens(
        self,
        generation_config: GenerationConfig,
        has_attention_mask: bool,
        device: torch.device,
    ) -> None:
        # Ensure special tokens are set to default if missing.
        generation_config.bos_token_id = generation_config.bos_token_id if generation_config.bos_token_id is not None else 1025
        generation_config.eos_token_id = generation_config.eos_token_id if generation_config.eos_token_id is not None else 1024
        generation_config.pad_token_id = generation_config.pad_token_id if generation_config.pad_token_id is not None else 1024
        generation_config.decoder_start_token_id = generation_config.decoder_start_token_id if generation_config.decoder_start_token_id is not None else 1025
        
        # Convert to tensors on the correct device.
        generation_config._bos_token_tensor = torch.tensor([generation_config.bos_token_id], device=device)
        generation_config._pad_token_tensor = torch.tensor([generation_config.pad_token_id], device=device)
        generation_config._eos_token_tensor = torch.tensor([generation_config.eos_token_id], device=device)
        generation_config._decoder_start_token_tensor = torch.tensor([generation_config.decoder_start_token_id], device=device)

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        generation_config: GenerationConfig,
        model_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        # Flatten inputs to shape: (batch_size * num_codebooks, seq_len)
        inputs = inputs.view(-1, inputs.shape[-1])
        batch_size, seq_len = inputs.shape

        attention_mask = torch.ones((batch_size, seq_len), 
                                    device=inputs.device, 
                                    dtype=torch.int64)

        if generation_config.pad_token_id is not None:
            # If pad_token_id is a list, take the first element
            pad_token_id = generation_config.pad_token_id[0] if isinstance(generation_config.pad_token_id, list) else generation_config.pad_token_id
            pad_mask = (inputs != pad_token_id).to(torch.int64)
            attention_mask = attention_mask * pad_mask

        if generation_config.eos_token_id is not None:
            eos_token_id = generation_config.eos_token_id[0] if isinstance(generation_config.eos_token_id, list) else generation_config.eos_token_id
            eos_mask = (inputs != eos_token_id).to(torch.int64)
            attention_mask = attention_mask * eos_mask

        return attention_mask

    def _prepare_generated_length(
        self,
        generation_config: GenerationConfig,
        has_default_max_length: bool,
        has_default_min_length: bool,
        model_input_name: str,
        inputs_tensor: torch.Tensor,
        input_ids_length: int,
    ) -> GenerationConfig:
        generation_config = copy.deepcopy(generation_config)
        if has_default_max_length and generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        if has_default_min_length and generation_config.min_new_tokens is not None:
            generation_config.min_length = generation_config.min_new_tokens + input_ids_length
        if generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: min_length ({generation_config.min_length}) cannot be larger than "
                f"max_length ({generation_config.max_length})"
            )
        return generation_config

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.Tensor,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        device: torch.device = None,
    ) -> LogitsProcessorList:
        processor_list = logits_processor if logits_processor is not None else LogitsProcessorList()

        # All processors should be instances of LogitsProcessor
        for processor in processor_list:
            if not hasattr(processor, "__call__"):
                raise ValueError(f"Processor {processor} has no __call__ method")

        if generation_config.min_length is not None and generation_config.min_length > 0:
            # Ensure eos_token_id is a tensor on the correct device
            if isinstance(generation_config.eos_token_id, (int, list)):
                eos_token_id = torch.tensor(generation_config.eos_token_id, device=device)
            else:
                eos_token_id = generation_config.eos_token_id.to(device)
                
            processor_list.append(MinLengthLogitsProcessor(generation_config.min_length, eos_token_id))

        return processor_list

    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].index_select(0, expanded_return_idx)

        if model_kwargs.get("delay_pattern_mask") is not None:
            model_kwargs["delay_pattern_mask"] = model_kwargs["delay_pattern_mask"].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is not None:
                model_kwargs["encoder_outputs"] = BaseModelOutput(
                    last_hidden_state=model_kwargs["encoder_outputs"].last_hidden_state.index_select(0, expanded_return_idx)
                )
        return input_ids, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initialize input_ids for generation if needed."""
        if inputs is not None:
            return inputs

        if model_kwargs is not None and "encoder_outputs" in model_kwargs:
            shape = model_kwargs["encoder_outputs"][0].size()[:-1]  
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        batch_size = 1
        if model_kwargs is not None:
            for value in model_kwargs.values():
                if isinstance(value, torch.Tensor):
                    batch_size = value.shape[0]
                    break

        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
    ) -> StoppingCriteriaList:
        criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            if max_position_embeddings is not None and generation_config.max_length > max_position_embeddings:
                logger.warning(
                    f"You're trying to generate more tokens ({generation_config.max_length}) than the specified maximum "
                    f"length ({max_position_embeddings}). This will result in the model generating less tokens than requested."
                )
            criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        return criteria

    @torch.no_grad()
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        generation_config: Optional[GenerationConfig] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[torch.LongTensor, StoppingCriteriaList]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config is None:
            generation_config = self.generation_config
        
        # init sequence length tensors
        sequence_lengths = torch.ones(input_ids.shape[0], device=input_ids.device)
        unfinished_sequences = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.long)

        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.last_hidden_states[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            
            # sample next token from distribution
            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores / generation_config.temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # update generated ids, model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            if streamer is not None:
                streamer.put(next_tokens.cpu())
                
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False, standardize_cache_format=True
            )

            # if eos token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != generation_config.eos_token_id).long()
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
                
            # increase cur_len
            sequence_lengths = sequence_lengths + 1

            # Check stopping criteria
            should_stop = False
            for criteria in stopping_criteria:
                should_stop = criteria(input_ids, None).any()
                if should_stop:
                    break
            if should_stop:
                break

        if streamer is not None:
            streamer.end()

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            
        return model_kwargs

class ConditionalGenerationModel(nn.Module):
    config_class = ModelConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[CausalLM] = None,
    ):
        if config is None and (text_encoder is None or audio_encoder is None or decoder is None):
            raise ValueError("Either a configuration has to be provided, or all three of text encoder, audio encoder and decoder.")
        if config is None:
            config = ModelConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.text_encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the text encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.text_encoder.hidden_size} for"
                    " `config.text_encoder.hidden_size`."
                )

        super().__init__(config)
        if text_encoder is None:
            from transformers.models.auto.modeling_auto import AutoModelForTextEncoding
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)

        if audio_encoder is None:
             audio_encoder = XCodecModel()

        if decoder is None:
            decoder = CausalLM(config.decoder)

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder

        if self.text_encoder.config.to_dict() != self.config.text_encoder.to_dict():
            logger.warning(
                f"Config of the text_encoder: {self.text_encoder.__class__} is overwritten by shared text_encoder config:"
                f" {self.config.text_encoder}"
            )
        #if self.audio_encoder.config.to_dict() != self.config.audio_encoder.to_dict():
        #    logger.warning(
        #        f"Config of the audio_encoder: {self.audio_encoder.__class__} is overwritten by shared audio_encoder config:"
        #        f" {self.config.audio_encoder}"
        #    )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )


        self.text_encoder.config = self.config.text_encoder 
        #self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder

        self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(f"The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head")

        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError("The selected decoder is not prepared for the encoder hidden states to be passed.")
        self.post_init()
        self.tie_weights()

    def post_init(self):
        self._tp_plan = getattr(self.config, "base_model_tp_plan", None)
        self._pp_plan = getattr(self.config, "base_model_pp_plan", None)

    def tie_weights(self):
        if self.config.tie_encoder_decoder:
            decoder_base_model_prefix = self.decoder.base_model_prefix
            tied_weights = self._tie_encoder_decoder_weights(
                self.text_encoder,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix,
                "text_encoder",
            )
            self._dynamic_tied_weights_keys = tied_weights

    def _init_weights(self, module):
        std = self.decoder.config.initializer_factor
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    def get_audio_encoder(self):
        return self.audio_encoder

    def get_text_encoder(self):
        return self.text_encoder

    def get_encoder(self):
        return self.get_text_encoder()

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_sliding_window_flex_mask: Optional[torch.Tensor] = None,
        decoder_fine_selection_flex_mask: Optional[torch.Tensor] = None,
        decoder_cache: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if input_ids is not None:
                encoder_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
                if not return_dict:
                    encoder_hidden_states = encoder_outputs[0]
                else:
                    encoder_hidden_states = encoder_outputs.last_hidden_state
                    
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

                if attention_mask is not None:
                    encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]
            
            else:
                raise ValueError("Either encoder_outputs or input_ids must be provided.")
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
            
            
        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            ).transpose(1, 2)

        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            audio_encoder_outputs = self.audio_encoder(
                input_values=input_values,
                padding_mask=padding_mask,
                **kwargs,
            )
            audio_codes = audio_encoder_outputs if isinstance(audio_encoder_outputs, torch.Tensor) else audio_encoder_outputs
            frames, bsz, codebooks, seq_len = audio_codes.shape

            if frames != 1:
                raise ValueError(
                    f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                    "disabled by setting `chunk_length=None` in the audio encoder."
                )

            if self.config.decoder.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
                audio_codes = audio_codes.repeat_interleave(2, dim=2)

            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            position_ids=decoder_position_ids,
            sliding_window_flex_mask=decoder_sliding_window_flex_mask,
            fine_selection_flex_mask=decoder_fine_selection_flex_mask,
            cache=decoder_cache,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        if not return_dict:
            return decoder_outputs + (encoder_hidden_states,)

        return Seq2SeqOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            decoder_router_logits=decoder_outputs.router_logits,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            codebook_losses=decoder_outputs.codebook_losses,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_delay_pattern_mask=None,
        guidance_scale=None,
        cache_position=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                bos_token_id=self.generation_config.bos_token_id,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

        if guidance_scale is not None and guidance_scale > 1:
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))


        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device)
        elif use_cache:
            cur_len = decoder_input_ids.shape[1]
            cache_position = cache_position[-cur_len:]

        if decoder_attention_mask is None: #and prompt_attention_mask is not None:
            input = decoder_input_ids.reshape(-1, self.decoder.num_codebooks, decoder_input_ids.shape[-1])
            bsz, _, seq_len = input.shape
            input_shape = (bsz, seq_len)

            past_key_values_length = 0
            if cache_position is not None:
                past_key_values_length = cache_position[0]
            elif past_key_values is not None:
                past_key_values_length = past_key_values.get_seq_length()

            if past_key_values is None or (isinstance(past_key_values, EncoderDecoderCache) and past_key_values.get_seq_length() == 0):
                decoder_attention_mask = torch.ones(input_shape, device=self.device, dtype=decoder_input_ids.dtype)

        res = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids.contiguous(),
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "inputs_embeds": inputs_embeds,
        }
        return res


    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = (
            torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device)
            * decoder_start_token_id
        )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat((torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),dim=-1,)
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask
        return decoder_input_ids, model_kwargs


    def _prepare_text_encoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get text encoder
        encoder = self.get_text_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        guidance_scale = generation_config.guidance_scale

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state

        # we optionnally project last_hidden_state to avoid recomputing every time
        encoder_hidden_states = last_hidden_state
        if (
            self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if guidance_scale is not None and guidance_scale > 1:
            last_hidden_state = torch.concatenate([last_hidden_state, torch.zeros_like(last_hidden_state)], dim=0)
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = torch.concatenate(
                    [model_kwargs["attention_mask"], torch.zeros_like(model_kwargs["attention_mask"])], dim=0
                )

        if model_kwargs["attention_mask"] is not None:
            encoder_hidden_states = encoder_hidden_states * model_kwargs["attention_mask"][..., None]

        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        return model_kwargs

    def _prepare_audio_encoder_kwargs_for_generation(
        self, input_values, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get audio encoder
        encoder = self.get_audio_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
        encoder_kwargs["return_dict"] = True

        if self.decoder.config.audio_channels == 1:
            encoder_kwargs[model_input_name] = input_values
            audio_encoder_outputs = encoder.encode(**encoder_kwargs)
            audio_codes = audio_encoder_outputs.audio_codes
            audio_scales = audio_encoder_outputs.audio_scales
            frames, bsz, codebooks, seq_len = audio_codes.shape
        else:
            if input_values.shape[1] != 2:
                raise ValueError(f"Expected stereo audio (2-channels) but example has {input_values.shape[1]} channel.")
            encoder_kwargs[model_input_name] = input_values[:, :1, :]
            audio_encoder_outputs_left = encoder.encode(**encoder_kwargs)
            audio_codes_left = audio_encoder_outputs_left.audio_codes
            audio_scales_left = audio_encoder_outputs_left.audio_scales

            encoder_kwargs[model_input_name] = input_values[:, 1:, :]
            audio_encoder_outputs_right = encoder.encode(**encoder_kwargs)
            audio_codes_right = audio_encoder_outputs_right.audio_codes
            audio_scales_right = audio_encoder_outputs_right.audio_scales

            frames, bsz, codebooks, seq_len = audio_codes_left.shape
            audio_codes = audio_codes_left.new_ones((frames, bsz, 2 * codebooks, seq_len))

            audio_codes[:, :, ::2, :] = audio_codes_left
            audio_codes[:, :, 1::2, :] = audio_codes_right

            if audio_scales_left != [None] or audio_scales_right != [None]:
                audio_scales = torch.stack([audio_scales_left, audio_scales_right], dim=1)
            else:
                audio_scales = [None] * bsz

        if frames != 1:
            raise ValueError(
                f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                "disabled by setting `chunk_length=None` in the audio encoder."
            )

        decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

        model_kwargs["decoder_input_ids"] = decoder_input_ids
        model_kwargs["audio_scales"] = audio_scales
        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.decoder.pad_token_id, self.config.decoder.bos_token_id).transpose(1, 2)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _get_decoder_start_token_id(
        self, decoder_start_token_id: Union[int, List[int]] = None, bos_token_id: int = None
    ) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError("`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.")

    def _get_cache(self, cache_implementation: str, max_batch_size: int, max_cache_len: int, model_kwargs) -> Cache:
        cache_cls: Cache = NEED_SETUP_CACHE_CLASSES_MAPPING[cache_implementation]
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        if hasattr(self, "_cache"):
            cache_to_check = self._cache.self_attention_cache if requires_cross_attention_cache else self._cache

        if cache_implementation == "sliding_window":
            max_cache_len = min(self.config.sliding_window, max_cache_len)

        need_new_cache = (
            not hasattr(self, "_cache")
            or (not isinstance(cache_to_check, cache_cls))
            or cache_to_check.max_batch_size != max_batch_size
            or cache_to_check.max_cache_len < max_cache_len
        )

        if requires_cross_attention_cache and hasattr(self, "_cache"):
            need_new_cache = (
                need_new_cache
                or self._cache.cross_attention_cache.max_cache_len != model_kwargs["encoder_outputs"][0].shape[1]
            )

        if need_new_cache:
            if hasattr(self.config, "_pre_quantization_dtype"):
                cache_dtype = self.config._pre_quantization_dtype
            else:
                cache_dtype = self.dtype
            cache_kwargs = {
                "config": self.config.decoder,
                "max_batch_size": max_batch_size,
                "max_cache_len": max_cache_len,
                "device": self.device,
                "dtype": cache_dtype,
            }
            self._cache = cache_cls(**cache_kwargs)
            if requires_cross_attention_cache:
                encoder_kwargs = cache_kwargs.copy()
                encoder_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
                config_cross_attention_cache = copy.deepcopy(self.config.decoder)
                config_cross_attention_cache.update(
                    {"num_key_value_heads": self.config.decoder.num_cross_attention_key_value_heads}
                )
                encoder_kwargs["config"] = config_cross_attention_cache
                self._cache = EncoderDecoderCache(self._cache, cache_cls(**encoder_kwargs))
        else:
            self._cache.reset()
        return self._cache

    def freeze_encoders(self, freeze_text_encoder=True):
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder._requires_grad = False

        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder._requires_grad = False

        if self.mert_model is not None:
            for param in self.mert_model.parameters():
                param.requires_grad = False
            self.mert_model._requires_grad = False

    def freeze_cross(self, freeze_cross = True):
        if freeze_cross:
            logger.info("Frozen the following layers:")
            logger.info("- text_encoder")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder._requires_grad = False

            if hasattr(self, 'enc_to_dec_proj'):
                logger.info("- enc_to_dec_proj")
                for param in self.enc_to_dec_proj.parameters():
                    param.requires_grad = False
                self.enc_to_dec_proj._requires_grad = False

            if hasattr(self, 'mert_model'):
                logger.info("- mert_model")
                for param in self.mert_model.parameters():
                    param.requires_grad = False
                self.mert_model._requires_grad = False

            if hasattr(self, 'mert_layer_agg'):
                logger.info("- mert_layer_agg")
                for param in self.mert_layer_agg.parameters():
                    param.requires_grad = False
                self.mert_layer_agg._requires_grad = False

            if hasattr(self, 'audio_mert_to_dec_proj'):
                logger.info("- audio_mert_to_dec_proj")
                for param in self.audio_mert_to_dec_proj.parameters():
                    param.requires_grad = False
                self.audio_mert_to_dec_proj._requires_grad = False

            if hasattr(self, 'prompt_cross_attention') and self.prompt_cross_attention:
                if hasattr(self, 'embed_prompts'):
                    logger.info("- embed_prompts")
                    for param in self.embed_prompts.parameters():
                        param.requires_grad = False
                    self.embed_prompts._requires_grad = False


    def freeze_embed_prompts(self, freeze_embed_prompts=True):
        if freeze_embed_prompts:
            for param in self.embed_prompts.parameters():
                param.requires_grad = False
            self.embed_prompts._requires_grad = False

            for param in self.decoder.model.decoder.embed_tokens.parameters():
                param.requires_grad = False
            self.decoder.model.decoder.embed_tokens._requires_grad = False


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) == tuple:
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])
        
        # 2. Set generation parameters if not already defined
        #logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        #stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([CustomLogitsProcessor(generation_config.eos_token_id, self.decoder.num_codebooks, batch_size, inputs_tensor.device)])
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if "encoder_outputs" not in model_kwargs:
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(
                model_kwargs["input_values"],
                model_kwargs,
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            bos_token_id=generation_config._bos_token_tensor,
            device=inputs_tensor.device,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError("This model does not support `cache_implementation='static'`.")
                if not self.prompt_cross_attention:
                    input_embeds_seq_length = model_kwargs["inputs_embeds"].shape[1]
                    max_cache_len = generation_config.max_length + input_embeds_seq_length - input_ids_length
                else:
                    max_cache_len = self.generation_config.max_length

                model_kwargs["past_key_values"] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    max_cache_len,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                raise ValueError("This model does not support the quantized cache. If you want your model to support quantized ")

        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get("past_key_values", None)
            requires_cross_attention_cache = (self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None)
            if past is None:
                model_kwargs["past_key_values"] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
            elif isinstance(past, tuple):
                model_kwargs["past_key_values"] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )

        # build the delay pattern mask for offsetting each codebook prediction by 1
        delayed_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config._bos_token_tensor,
            pad_token_id=generation_config._pad_token_tensor,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        # delayed_input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(delayed_input_ids.cpu())

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()
        logger.info(f'generation_mode: {generation_mode}')

        # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
        #if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
        #    logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
        #    generation_config.guidance_scale = None


        # 9. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=delayed_input_ids.device,
        )

        # 10. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # expand input_ids with `num_return_sequences` additional sequences per batch
            delayed_input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=delayed_input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 11. run sample
            outputs = self._sample(
                delayed_input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        #output_ids.shape [bsz*num_codebooks, seq_len]. Apply the pattern mask to the final ids
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # Revert the pattern delay mask by filtering the eos and bos token ids from the delay pattern mask
        _, mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=output_ids.shape[1],
        )

        mask = (mask != generation_config.bos_token_id) & (mask != generation_config.pad_token_id)
        output_ids = output_ids[mask].reshape(batch_size, self.decoder.num_codebooks, -1)

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...]

        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size

        if self.decoder.config.audio_channels == 1:
            output_values = self.audio_encoder.decode(
                output_ids,
                audio_scales=audio_scales,
            ).audio_values
            output_lengths = [audio.shape[0] for audio in output_values]
        else:
            codec_outputs_left = self.audio_encoder.decode(output_ids[:, :, ::2, :], audio_scales=audio_scales)
            output_values_left = codec_outputs_left.audio_values

            codec_outputs_right = self.audio_encoder.decode(output_ids[:, :, 1::2, :], audio_scales=audio_scales)
            output_values_right = codec_outputs_right.audio_values
            output_lengths = [audio.shape[0] for audio in output_values]
            output_values = torch.cat([output_values_left, output_values_right], dim=1)

        if generation_config.return_dict_in_generate:
            outputs["audios_length"] = output_lengths
            outputs.sequences = output_values
            return outputs
        else:
            return output_values