
# Self-Powered LLM Modality Expansion for Large Speech-Text Models

Tengfei Yu, Xuebo Liu, Zhiyi Hou, Liang Ding, Dacheng Tao, Min Zhang

**Harbin Institute of Technology**

**The University of Sydney**

**Nanyang Technological University**

<a href='https://github.com/ytf-philp/Self-powered-LSM'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'> </a>  <a href=''><img src='https://img.shields.io/badge/SALMONN_13B-Demo-blue'></a>
<a href=''><img src='https://img.shields.io/badge/SALMONN_7B-Demo-orange'></a>

## 👀 Overview

We introduce **Self-Powered LSM**, a system that leverages self-powered data to enhance the speech modality capabilities of Large Language Models (LLMs).

<div align="left">
  <img src="https://github.com/ytf-philp/Self-powered-LSM/blob/master/fig/image.png" width="70%">
</div>


## 🔥 News

- [2023-10-08] ✨ We have released [**the model checkpoint**](https://huggingface.co/tsinghua-ee/SALMONN)!
- [2024-04-07] 🤖 We have released all the codes you need to train your own self-powered LSM!


## Speech Instructional Dataset

We use Vicuna as the backbone LLM to generate our dataset. To utilize this dataset, download Librispeech-960, Common Voice 4.0, and Gigaspeech-L, and place them in `/data`.

Then use the speech instructional dataset to train the model. You can find the JSONL file [here](https://drive.google.com/file/d/1vrq9hA5dSLEv-_6Qm9kzHdbrxGjIXlng/view)

Additionally, you can generate your own self-powered data by running:

 ```
 bash ./self-powered/src/data_process/generate.sh
 ```

### 🌟 Structure

The model architecture of Self-Powered LSM is shown below. A window-level Q-Former is used as the connection module to fuse the outputs from a Whisper speech encoder and a BEATs audio encoder as augmented audio tokens, which are aligned with the LLM input space. Text prompt is used to instruct SALMONN to answer open-ended questions about the general audio inputs and the answers are in the LLM text responses. 

<div align=center><img src="resource/structure.png" height="100%" width="75%"/></div>

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
