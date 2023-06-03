# Vistaar: Diverse Benchmarks and Training sets for Indian Language ASR

 Vistaar is a set of 59 benchmarks and training datasets across various language and domain combinations such as news, education, literature, tourism etc. The training datasets are avaialable for 12 Indian languages amounting to over 10,700 hours of labelled audio data. We also train IndicWhisper models by fine-tuning the Whisper models on the Vistaar train dataset and observe that it has the lowest WER on 39 out of 59 Vistaar benchmarks.
 
## Benchmarks
Vistaar consists of benchmarks from several public datasets - Kathbath, FLEURS, CommonVoice, IndicTTS, MUCS, GramVaani across 12 languages. We evaluate IndicWhisper on these benchmarks along with 3 publicly available ASR systems and 2 commercially available systems. Below mentioned are the results

| Datasets      | bn       | gu       | hi       | kn       | ml       | mr       | or       | pa       | sa   | ta       | te       | ur       | avg   |
|---------------|----------|----------|----------|----------|----------|----------|----------|----------|------|----------|----------|----------|-------|
| Kathbath      | 16.6     | 17.8     | **10.3** | **19.3** | **34.8** | **19.9** | 24.7     | 16.9     | 45.6 | **24.2** | 25       | **11.9** | 22.3  |
| Kathbath Hard | 19.4     | **20.6** | **12.0** | **22.2** | **38.4** | **22.1** | 29.1     | 19.7     | 50.5 | **27.5** | 27.8     | **14.7** | 25.3  |
| CommonVoice   | 24.7     | -        | **11.4** | -        | **44.5** | 22.8     | **35.2** | 22.4     | -    | **29.2** | -        | 31.7     | 27.8  |
| FLEURS        | **20.9** | 23.5     | **15.0** | **18.6** | **22.6** | **20.5** | **32.9** | **23.1** | -    | **25.2** | **25.4** | **19.2** | 22.5  |
| IndicTTS      | **18.8** | **19.1** | **7.6**  | **13.2** | **21.4** | **11.4** | **15.0** | -        | -    | **17.2** | 33.8     | -        | 17.5  |
| MUCS          | -        | 33.2     | **12.0** | -        | -        | **12.8** | **27.5** | -        | -    | 28.3     | 32.1     | -        | 24.3  |
| Gramvaani     | -        | -        | **26.8** | -        | -        | -        | -        | -        | -    | -        | -        | -        | 26.8  |
| Average       | 20.1     | 22.8     | 13.6     | 18.3     | 32.3     | 18.2     | 27.4     | 20.5     | 48   | 25.3     | 28.8     | 19.4     | 24.6  |

*Word error rates (%) of IndicWhisper on the Vistaar Benchmark. Values in bold indicates benchmarks where IndicWhisper has the lowest WER.*


| Model         | Kathbath | Kathbath-Hard | FLEURS   | CommonVoice | IndicTTS | MUCS         | Gramvaani | Average   |
|---------------|----------|---------------|----------|-------------|----------|--------------|-----------|-----------|
| Google STT    | 14.3     | 16.7          | 19.4     | 20.8        | 18.3     | 17.8         | 59.9      | 23.9      |
| IndicWav2vec  | 12.2     | 16.2          | 18.3     | 20.2        | 15       | 22.9         | 42.1      | 21        |
| Azure STT     | 13.6     | 15.1          | 24.3     | 14.6        | 15.2     | 15.1         | 42.3      | 20        |
| Nvidia-medium | 14       | 15.6          | 19.4     | 20.4        | 12.3     | 12.4         | 41.3      | 19.4      |
| Nvidia-large  | 12.7     | 14.2          | 15.7     | 21.2        | 12.2     | **11.8**     | 42.6      | 18.6      |
| IndicWhisper  | **10.3** | **12.0**      | **11.4** | **15.0**    | **7.6**  | 12           | **26.8**  | **13.6**  |

*Comparison of publicly-available models on the Hindi subset of the Vistaar benchmark*

## Table of contents
- [Vistaar](#vistaar-diverse-benchmarks-and-training-sets-for-indian-language-asr)
  - [Benchmarks](#benchmarks)
  - [Table of contents](#table-of-contents)
  - [Resources](#resources)
    - [Download Training Datasets and Benchmarks](#download-training-datasets-and-benchmarks)
    - [Download Models](#download-models)
  - [Tutorials](#tutorials)
    - [Setting up your environment](#setting-up-your-environment)
    - [Evaluating ASR models](#evaluating-asr-models)
      - [Manifest creation](#manifest-creation)
      - [Running evaluation](#running-evaluation)
    - [Inference using IndicWhisper](#inference-using-indicwhisper)
      - [Sample structure of manifest file](#sample-structure-of-manifest-file)
      - [Running inference](#running-infernece)
    - [Training on Vistaar Train Datasets](#training-on-vistaar-train-datasets)
      - [Manifest creation](#manifest-creation)
      - [Running training](#running-training)
  - [Cite](#cite)
  - [License](#license)
  - [Contributors](#contributors)
  - [Contact](#contact)
## Resources

### Download Training Datasets and Benchmarks
|Datasets | Training Datasets | Benchmarks |
| - | - | - |
| Kathbath | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/kathbath.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/kathbath.zip) |
| Kathbath Hard | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/kathbath_noisy.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/kathbath_noisy.zip) |
| CommonVoice | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/commonvoice.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/commonvoice.zip) |
| FLEURS | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/fleurs.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/fleurs.zip) |
| IndicTTS | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/indictts.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/indictts.zip) |
| MUCS | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/mucs.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/mucs.zip) |
| gramvaani | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/gramvaani.zip) | [link](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/gramvaani.zip) |

### Download Models
|Language |Model Checkpoint |
| - | - | 
| Bengali | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/bengali_models.zip) |
| Gujarati | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/gujarati_models.zip) |
| Hindi | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/hindi_models.zip) |
| Kannada | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/kannada_models.zip) |
| Malayalam | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/malayalam_models.zip) |
| Marathi | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/marathi_models.zip) |
| Odia | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/odia_models.zip) |
| Punjabi | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/punjabi_models.zip) |
| Sanskrit | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/sanskrit_models.zip) |
| Tamil | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/tamil_models.zip) |
| Telugu | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/telugu_models.zip) |
| Urdu | [hf](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/all_lang_models/urdu_models.zip) |

## Tutorials
### Setting up your environment
- Setting up python virtual environment
  ```
  python -m venv <env_name>
  source <env_name>/bin/activate
  ```
- Installing/Updating Libraries
  ```
  sudo apt install ffmpeg
  pip install -r requirements.txt
  ```
### Evaluating ASR models
  - Manifest creation
    - For each dataset Download and extract the benchmark data in a directory. The data should be extracted in such a way that each folder inside should contain data for a particular language i.e each language specific folder should contain train, valid and test folder and within them the audio + transcript.txt 
    - Sample structure of folder tree:
    ```
      kathbath
          ├── bengali
          │   ├── audio
          │   └── transcript.txt
          │
          └── gujarti
              ├── audio
              └── transcript.txt 
          .
          .
          .
          .
    ```
    - The manifest needs to be a Json file where each line is a dictionary with audio filepath, duration of audio and text transcript
    ```
     {"audio_filepath": <path to audio file 1>, "duration": <duration of audio file 1>, "text": <transcript of audio file 1>}
     {"audio_filepath": <path to audio file 2>, "duration": <duration of audio file 2>, "text": <transcript of audio file 2>}
     .
     .
     .
    ```
  - Running evaluation
    ```
    python evaluation.py --model_path=<model path> \
    --manifest_path=<manifest path in vistaar> \
    --manifest_name=<dataset name in vistaar> \
    --device=<gpu to use> \
    --batch_size=<batch size> \
    --language=<2 letter language code>
    ```
### Inference using IndicWhisper
  - Sample structure of manifest file
  ```
  {"audio_filepath":<path to audio file 1>}
  {"audio_filepath":<path to audio file 2>}
  .
  .
  .
  ```
  - Running inference
  ```
  deepspeed --include localhost:<gpus to include> \
  transcribe.py <manifest path> \
  <model path> \
  <current language> \
  <batch size>
  <output path>
  ```
### Training on Vistaar Train Datasets
  - Manifest creation
    - Follow the steps as in [Evaluating ASR models](#evaluating-asr-models) for the vistaar training datasets
  - Running training
  ```
  deepspeed --include localhost:<gpus to include> training.py \
  --deepspeed=<path to deepspeed config file> \
  --model_name_or_path=<model path> \
  --dataset_name=<dataset language directory path> \
  --language=<language> \
  --train_split_name=<train manifest name> \
  --eval_split_name=<validation manifest name> \
  --max_steps="5000" \
  --output_dir=<output directory path> \
  --cache_dir=<cache directory for downloaded models> \
  --per_device_train_batch_size="64" \
  --per_device_eval_batch_size="32" \
  --gradient_accumulation_steps="1" \
  --logging_steps="10" \
  --learning_rate="1e-5" \
  --warmup_steps="500" \
  --evaluation_strategy="steps" \
  --eval_steps="500" \
  --save_strategy="steps" \
  --save_steps="500" \
  --generation_max_length="225" \
  --length_column_name="input_length" \
  --max_duration_in_seconds="30" \
  --text_column_name="sentence" \
  --freeze_feature_encoder="False" \
  --report_to="tensorboard" \
  --metric_for_best_model="wer" \
  --greater_is_better="False" \
  --load_best_model_at_end \
  --gradient_checkpointing \
  --fp16 \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --do_normalize_eval="False" \
  --streaming="True" \
  --use_auth_token="True" \
  --push_to_hub="True"
  ```
