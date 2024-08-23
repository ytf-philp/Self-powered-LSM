
# Self-Powered LLM Modality Expansion for Large Speech-Text Models

Tengfei Yu, Xuebo Liu, Zhiyi Hou, Liang Ding, Dacheng Tao, Min Zhang


<a href='https://github.com/ytf-philp/Self-powered-LSM'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'> </a>  

## 👀 Overview

**Self-Powered LSM** is a pioneering Large Speech-Text Model that utilizes self-generated data to boost the speech capabilities of Large Language Models.

<div align="left">
  <img src="https://github.com/ytf-philp/Self-powered-LSM/blob/master/fig/image.png" width="70%">
</div>


## 🔥 News

- [2024-08-20] 🤖 Release of all necessary code for training your own Self-Powered LSM!


## Speech Instructional Dataset

We use Vicuna as the backbone LLM to generate dataset. To utilize our dataset, download Librispeech-960, Common Voice 4.0, and Gigaspeech-L, and place them in `/data`.

Then use the speech instructional dataset to train the model. You can find the JSONL file [here](https://drive.google.com/file/d/1vrq9hA5dSLEv-_6Qm9kzHdbrxGjIXlng/view)

Additionally, you can generate your own self-powered data by running:

 ```
 bash ./src/data_process/generate.sh
 ```

### 🌟 Model Structure

The model architecture of the Self-Powered LSM is depicted as follows: We use the encoder component of Whisper as the speech encoder and employ Vicuna-7B-1.5 as the large language model. 
Q-Former, serving as the connection module. The output sequence, integrated with the text instruction, is then fed into the LLM to generate the text response.

<div align=left><img src="fig/main.png" height="40%" width="35%"/></div>

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


## License Agreement

Researchers and developers are welcome to utilize our code for both academic and commercial purposes. Our models leverage the LLaMA-2 and Whisper architectures. We kindly ask that all users adhere to the MIT License for Whisper and the specific licensing requirements for LLaMA-2
<br>

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
```
<br>

## Contact Us

If you have any questions related to the code or the paper, feel free to email Tengfei Yu (921692739@qq.com).
