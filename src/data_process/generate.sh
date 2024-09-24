


export CUDA_VISIBLE_DEVICES=0

python asr_text_generation_unified_continue.py --manifest self-powered/augmentation/raw_all/output_part_repeat.jsonl --cate repeat --start 97268 --nshard 4 --rank 0 > /self-powered/src/data_process/new_augment/log/repeat1.log 2<&1 &

export CUDA_VISIBLE_DEVICES=1

python asr_text_generation_unified_repeat.py --manifest self-powered/augmentation/raw_all/output_part_repeat.jsonl --cate repeat --start 97268 --nshard 4 --rank 1 > /self-powered/src/data_process/new_augment/log/repeat2.log 2<&1 &

export CUDA_VISIBLE_DEVICES=2

python asr_text_generation_unified_repeat.py --manifest self-powered/augmentation/raw_all/output_part_repeat.jsonl --cate repeat --start 97268 --nshard 4 --rank 2 > /self-powered/src/data_process/new_augment/log/repeat3.log 2<&1 &

export CUDA_VISIBLE_DEVICES=3

python asr_text_generation_unified_repeat.py --manifest self-powered/augmentation/raw_all/output_part_repeat.jsonl --cate repeat --start 97268 --nshard 4 --rank 3 > /self-powered/src/data_process/new_augment/log/repeat4.log 2<&1 &

'''

Repeat the provided text, ensuring to maintain its original meaning and details
Rephrase the text without altering its initial intent and key information
Paraphrase the provided text while preserving all original facts and nuances
Echo the content of the text, maintaining its exact purpose and details
Retell the given information without changing its meaning or losing any critical data


Extract the most frequently occurring words or phrases in the text, excluding common stopwords, to identify main topics
Identify and list the most common words or phrases from the text, omitting typical stopwords, to highlight central themes
Determine the key words or phrases frequently used in the text, removing all usual stopwords, to discern the main topics
Find the recurring words or phrases in the text, ignoring common stopwords, to ascertain the primary themes
Extract significant words or phrases that appear often in the text, exclude basic stopwords, to uncover the main subjects

Determine the primary purpose of the speech and evaluate how clearly and effectively the message is conveyed
Identify the main intent of the speech and assess the clarity and effectiveness of its delivery
Ascertain the fundamental objective of the speech and critique the transparency and efficiency of its presentation
Figure out the chief aim of the speech and judge how lucidly and effectively the ideas are presented
Assess the central purpose of the speech and evaluate the directness and impact of its expression

Determine the sentiment of the text and identify which sections contribute most to sentiment
Analyze the overall mood of the text and pinpoint the parts that heavily influence the sentiment
Evaluate the emotional tone of the text and determine which segments primarily affect the sentiment
Identify the feeling conveyed by the text and specify which portions substantially shape this sentiment
Assess the sentiment expressed in the text and highlight which areas contribute most to this feeling

Please write a coherent and engaging Chinese continuation of the given English text with less than 50 words
Compose a logical and captivating follow-up in Chinese to the provided English text within 50 words
Craft a coherent and appealing Chinese extension for the English text, ensuring it does not exceed 50 words
Develop a consistent and attractive Chinese continuation of the English text, keeping it under 50 words
Write a fluent and engaging continuation in Chinese of the English text, limited to 50 words

Provide the translation from English to Chinese
Translate the given English content into Chinese
Render the English text into Chinese
Convert the specified English text into Chinese
Translate the provided English material into the Chinese language


Provide the English transcription according to the speech
Transcribe the spoken words into written English text
Convert the spoken English into a written transcript
Create a written transcription of the spoken English
Write down the English speech as a text transcript
'''