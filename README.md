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


## Updates

## Table of contents
- [Vistaar](#vistaar-diverse-benchmarks-and-training-sets-for-indian-language-asr)
  - [Benchmarks](#benchmarks)
  - [Table of contents](#table-of-contents)
  - [Resources](#resources)
    - [Download Models](#download-models)
    - [Hosted API Usage](#hosted-api-usage)
    - [Accessing on ULCA](#accessing-on-ulca)
  - [Quick start](#quick-start)
    - [Python Inference](#python-inference)
    - [Huggingface Inference](#huggingface-inference)
  - [Tutorials](#tutorials)
    - [Setting up your environment](#setting-up-your-environment)
    - [Pretraining](#pretraining)
      - [Data preparation](#data-preparation)
      - [Manifest Creation](#manifest-creation)
    - [Training procedure and code](#training-procedure-and-code)
    - [Finetuning](#finetuning)
      - [Data preparation](#data-preparation-1)
      - [Finetuning procedure and code](#finetuning-procedure-and-code)
      - [Finetuning procedure and code](#finetuning-procedure-and-code-1)
    - [Language Modelling (LM)](#language-modelling-lm)
      - [Data preparation](#data-preparation-2)
      - [Training details](#training-details)
    - [Evaluating ASR models](#evaluating-asr-models)
    - [Model exporting](#model-exporting)
    - [Deployment](#deployment)
  - [Cite](#cite)
  - [License](#license)
  - [Contributors](#contributors)
  - [Contact](#contact)




