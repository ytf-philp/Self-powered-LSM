
# Self-Powered LLM Modality Expansion for Large Speech-Text Models

Tengfei Yu, Xuebo Liu, Zhiyi Hou, Liang Ding, Dacheng Tao, Min Zhang

**Harbin Institute of Technology**

**The University of Sydney**

**Nanyang Technological University**

<a href='https://github.com/ytf-philp/Self-powered-LSM'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'> </a>  

## 👀 Overview

We introduce **Self-Powered LSM**, a Large Speech-Text Model that leverages self-powered data to enhance the speech modality capabilities of Large Language Models.

<div align="left">
  <img src="https://github.com/ytf-philp/Self-powered-LSM/blob/master/fig/image.png" width="70%">
</div>


## 🔥 News

- [2024-08-25] ✨ We have released [**the model checkpoint**]()!
- [2024-08-20] 🤖 We have released all the codes you need to train your own self-powered LSM!


## Speech Instructional Dataset

We use Vicuna as the backbone LLM to generate our dataset. To utilize this dataset, download Librispeech-960, Common Voice 4.0, and Gigaspeech-L, and place them in `/data`.

Then use the speech instructional dataset to train the model. You can find the JSONL file [here](https://drive.google.com/file/d/1vrq9hA5dSLEv-_6Qm9kzHdbrxGjIXlng/view)

Additionally, you can generate your own self-powered data by running:

 ```
 bash ./self-powered/src/data_process/generate.sh
 ```

### 🌟 Structure

The model architecture of the Self-Powered LSM is depicted as follows: A window-level Q-Former serves as the connecting module, integrating outputs from the Whisper speech encoder as enhanced audio tokens. These tokens are aligned with the input space of the LLM. A text prompt guides the LSM to address open-ended questions concerning general audio inputs, with responses generated in LLM text format. 

<div align=center><img src="fig/main.png" height="40%" width="40%"/></div>

### 🚀 Train Self-Powered LSM

**Preprocessing Data**
* Tokenize training dataset 
```
python ./self-powered/src/speech_text_paired_dataset_llama.py
```
**Training**
* To train the model, you may run:
```
bash ./self-powered/scripts/vicuna_small_sft.sh
```
**Evaluation**
* Tokenize the evaluation dataset
```
python ./self-powered/src/evaluate_token_asr.py
``` 
* Inference
```
bash self-powered/scripts/inference_ASR.sh $DATA $MODEL $SAVE_PATH
``` 


## License
* The license of our project is [Apache License 2.0]()
* Our models are based on Llama2 and Whisper. If you want to use our models, please do not violate the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) of whisper and the [License](https://github.com/facebookresearch/llama/blob/main/LICENSE) of LLaMA-2
