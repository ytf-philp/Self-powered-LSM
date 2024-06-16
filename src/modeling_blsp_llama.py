import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PreTrainedModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig, WhisperConfig,LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
try:
    from .configuration_blsp import SpeechLLMConfig
    from .Qformer import BertLMHeadModel,BertConfig
    from .modeling_whisper_encoder import WhisperEncoder
except:
    from  Qformer import BertLMHeadModel,BertConfig
    from configuration_blsp import SpeechLLMConfig
    from modeling_whisper_encoder import WhisperEncoder


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class BlspModel(PreTrainedModel):
    config_class = SpeechLLMConfig
    base_model_prefix = "blsp"
    def __init__(self, config: SpeechLLMConfig):
        super().__init__(config)
        
        print("Loading Whisper")
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.whisper_model = WhisperEncoder(self.whisper_config)
        
        print("Loading Llama")
        self.llama_config = LlamaConfig(**config.llama_config)
        self.llama_model = LlamaForCausalLM(self.llama_config)
        #self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_config)
        print("Loading Q-former")
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            config.speech_qformer_token_num,
            self.whisper_config.d_model,
            config.speech_qformer_layer,
        )
        self.second_per_frame =config.second_per_frame
        self.second_stride =config.second_stride
        in_d = self.whisper_config.d_model
        out_d = self.llama_config.hidden_size
        self.speech_llama_proj = nn.Linear(
            self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size)

        self.ln_speech = nn.LayerNorm(self.whisper_model.config.d_model)

    
    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speech_values: Optional[torch.FloatTensor] = None,
        speech_attention_mask: Optional[torch.LongTensor] = None,
        suffix_input_ids: Optional[torch.LongTensor] = None,
        suffix_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        instruct_labels: Optional[torch.LongTensor] = None,
        
    ):
        ### 1. forward speech
        speech_embeds, speech_attention_mask, B = self.get_speech_features(speech_values, speech_attention_mask)


        #speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        #speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        #speech_labels = torch.LongTensor(speech_embeds.size(0), speech_embeds.size(1)).fill_(-100).to(speech_embeds.device)
        
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)  #1376 9 768
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_attention_mask,
            return_dict=True,
        )

        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_attention_mask = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        speech_labels = torch.LongTensor(speech_embeds.size(0), speech_embeds.size(1)).fill_(-100).to(speech_embeds.device)

        ### 2. forward llama
        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds = torch.cat([prefix_embeds, speech_embeds,suffix_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, speech_attention_mask, suffix_attention_mask], dim=1)
        if labels is not None:
            labels = torch.cat([instruct_labels, speech_labels, labels], dim=1)
        
        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
        )


    def get_speech_features(self, speech_values, speech_attention_mask):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        speech_lengths = output.output_lengths
        speech_embeds = self.ln_speech(speech_embeds)
        B, T, C = speech_embeds.shape  # 16 781 768   1376 9 768
        kernel = round(T * self.second_per_frame / 30.0)
        stride = round(T * self.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        return speech_embeds, speech_atts, B

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        suffix_input_ids,
        speech_values=None,
        speech_attention_mask=None,
        generation_config=None
    ):
        inputs_embeds, attention_mask = [], []

        prefix_embeds = self.llama_model.get_input_embeddings()(input_ids)
        prefix_attns = torch.ones(prefix_embeds.size(0), prefix_embeds.size(1), dtype=torch.long).to(prefix_embeds.device)
        inputs_embeds.append(prefix_embeds)
        attention_mask.append(prefix_attns)

        if speech_values is not None:
            speech_embeds, speech_attention_mask,B = self.get_speech_features(speech_values, speech_attention_mask)
            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_attention_mask,
                return_dict=True,
            )

            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
            speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
            speech_attention_mask = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            inputs_embeds.append(speech_embeds)
            attention_mask.append(speech_attention_mask)
            
        suffix_embeds = self.llama_model.get_input_embeddings()(suffix_input_ids)
        suffix_attns = torch.ones(suffix_embeds.size(0), suffix_embeds.size(1), dtype=torch.long).to(suffix_embeds.device)
        inputs_embeds.append(suffix_embeds)
        attention_mask.append(suffix_attns)

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        attention_mask = torch.cat(attention_mask, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
    
    
        
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config=None
    ):
        inputs_embeds = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0]
                embeds = self.llama_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0], h[1]
                speech_embeds, _ = self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config
        )
