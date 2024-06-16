"""BLSP config"""

from transformers import PretrainedConfig, LlamaConfig, WhisperConfig, BertConfig
from transformers import logging

logger = logging.get_logger(__name__)

class SpeechLLMConfig(PretrainedConfig):
    def __init__(
        self, 
        whisper_config=None, 
        llama_config=None,
        qformer_config=None,
        speech_qformer_layer=2,
        speech_qformer_token_num=1,
        second_per_frame=0.33333,
        second_stride=0.33333,
        lora=True,
        lora_alpha=32,
        lora_rank=4,
        lora_dropout=0.1,
        do_train=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if whisper_config is None:
            whisper_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")
        
        if llama_config is None:
            llama_config = {}
            logger.info("llama config is None. Initializing the LlamaConfig with default values")
        
        '''if qformer_config is None:
            qformer_config = {}
            logger.info("q_former config is None. Initializing the QFormer Config with default values")'''
            
        self.whisper_config = WhisperConfig(**whisper_config).to_dict()
        self.llama_config = LlamaConfig(**llama_config).to_dict()
        #self.qformer_config = BertConfig(**qformer_config).to_dict()
        self.speech_qformer_layer = speech_qformer_layer
        self.speech_qformer_token_num = speech_qformer_token_num
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        self.lora=lora
        self.lora_alpha=lora_alpha
        self.lora_rank=lora_rank
        self.lora_dropout=lora_dropout
        self.do_train=do_train