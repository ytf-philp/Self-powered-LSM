import torch
from transformers import (
    AutoTokenizer,
    WhisperFeatureExtractor,
    AutoModelForCausalLM,
    GenerationConfig,
    WhisperConfig,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
from src.modeling_blsp import BlspModel
from src.modeling_whisper_encoder import WhisperEncoder
from src.configuration_blsp import SpeechLLMConfig

# 加载模型和配置
whisper_model_path = "self-powered/model/whisper-large-v2"
llama_model_path = "self-powered/model/vicuna-7b-v1.5"
tokenizer_path = "self-powered/model/vicuna-7b-v1.5"
output_path = "self-powered/model/whisper_large_7B"

whisper_model = WhisperEncoder.from_pretrained(whisper_model_path)
llama_model = LlamaForCausalLM.from_pretrained(llama_model_path)
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_path)

# 配置新的模型
config = SpeechLLMConfig(
    whisper_config=vars(whisper_model.config),
    llama_config=vars(llama_model.config)
)

new_model = BlspModel(config)
new_model.whisper_model = whisper_model
new_model.llama_model = llama_model

# 保存模型和相关文件
new_model.save_pretrained(output_path, safe_serialization=False)
tokenizer.save_pretrained(output_path)
extractor.save_pretrained(output_path)
