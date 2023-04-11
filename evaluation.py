import argparse

from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import librosa
import pandas as pd
#from datasets import Dataset
from torch.utils.data import Dataset
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import pyarrow as pa
import soundfile as sf
import jiwer
import os
import string
import re
import time
import torch

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

def normalize_sentence(sentence, lang_code):
    '''
    Perform NFC -> NFD normalization for a sentence and a given language
    sentence: string
    lang_code: language code in ISO format
    '''
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang_code)
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence

class eval_dataset(Dataset):
    
    def __init__(self):
        self.audios = []
        self.sents = []
        
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        return {"raw": self.audios[i]['array'], "sampling_rate":self.audios[i]['sampling_rate'], "reference":self.sents[i], 
                "path": self.audios[i]['path'], "duration": self.audios[i]['duration']}
    
    def fill_data(self, aud, sent):
        self.audios.append(aud)
        self.sents.append(sent)

def get_data(split):
    js_data = json.loads(split)
    aud = {}
    aud['path'] = js_data['audio_filepath'].replace('/nlsasfs/home/ai4bharat/', '/workspace/')
    y, sr = sf.read(aud['path'])
    aud['duration'] = js_data['duration']
    aud['array'] = y
    aud['sampling_rate'] = sr
    
    return (aud, js_data['text'])
    #return (aud['path'], js_data['text'])
    

def main(args):
    
    MANIFEST_BASE_FOLDER = '/workspace/ai4bharat-pr/speechteam/asr_datasets'
    MODEL_BASE_FOLDER = '/workspace/ai4bharat-pr/speechteam/whisper_sai'
    batch_size = args.batch_size
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=os.path.join(MODEL_BASE_FOLDER, args.model_id), device=args.device,
    )
    
    # Special case to handle odia since odia is not supported by whisper model
    if args.language == 'or':
        whisper_asr.model.config.forced_decoder_ids = (
            whisper_asr.tokenizer.get_decoder_prompt_ids(
                language=None, task="transcribe"
            )
        )
    else:
        whisper_asr.model.config.forced_decoder_ids = (
            whisper_asr.tokenizer.get_decoder_prompt_ids(
                language=args.language, task="transcribe"
            )
        )
    
    with open(os.path.join(MANIFEST_BASE_FOLDER, args.dataset + '.json'), 'r') as f:
        data = f.read()
        splits = data.split('\n')[:-1]
    
    da = Parallel(n_jobs=128)(delayed(get_data)(split) for split in tqdm(splits))
    # data_dict = {}
    # data_dict['audio'] = []
    # data_dict['sentence'] = []
    
    dataset = eval_dataset()
    for d in da:
        dataset.fill_data(d[0], d[1])

    hypothesis = []
    ground_truth = []
    
    
    open(MODEL_BASE_FOLDER + '/evaluation_scripts/predictions/' + args.model_id.replace('/','_')
         + '_' + args.dataset.replace('/','_') + '_predictions.json', 'w').close()
    
    st = time.time()
    
    for out in tqdm(whisper_asr(dataset, batch_size=batch_size), total=len(dataset)):
        
        hyp = out['text']
        ref = out['reference'][0]
        hyp = hyp.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
        ref = ref.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
        if args.language[:2] != 'ur':
            hyp = normalize_sentence(hyp, args.language[:2])
            ref = normalize_sentence(ref, args.language[:2])
        hyp = re.sub(' +', ' ', hyp)
        ref = re.sub(' +', ' ', ref)
        
        hypothesis.append(hyp)
        ground_truth.append(ref)
        res = {
            "audio_filepath":out['path'][0],
            "duration":out['duration'][0],
            "text":ref,
            "pred_text":hyp
        }
        with open(MODEL_BASE_FOLDER + '/evaluation_scripts/predictions/' + args.model_id.replace('/','_')
         + '_' + args.dataset.replace('/','_') + '_predictions.json', 'a') as f:
            json.dump(res, f)
            f.write('\n')
    
    et = time.time()
     
    data = {}
    data['model'] = args.model_id
    data['dataset'] = args.dataset
    data['language'] = args.language
    data['cer'] = jiwer.cer(ground_truth, hypothesis)
    data['time'] = (et-st)/60
    data['batch_size'] = args.batch_size
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    data.update(measures)
    print(data)
    
    with open('results.csv', 'a') as results_fp:
        print(','.join([str(v) for v in data.values()]), file=results_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'` for the English split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    args = parser.parse_args()

    main(args)