
# Self-Powered LLM Modality Expansion for Large Speech-Text Models

## 👀 Overview

We introduce **Self-Powered LSM**, a system that leverages self-powered data to enhance the speech modality capabilities of Large Language Models (LLMs).

<div align="left">
  <img src="https://github.com/ytf-philp/Self-powered-LSM/blob/master/fig/image.png" width="70%">
</div>

## Speech Instructional Dataset

We use Vicuna as the backbone LLM to generate our dataset. To utilize this dataset, download Librispeech-960, Common Voice 4.0, and Gigaspeech-l, and place them in `/data`.

 You can find the JSONL file at  [Here](https://drive.google.com/file/d/1vrq9hA5dSLEv-_6Qm9kzHdbrxGjIXlng/view)

Additionally, you can generate your own self-powered data by running:

 ```
 bash ./self-powered/src/data_process/generate.sh
 ```

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

