import argparse
from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import librosa
import pandas as pd
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

lang_to_code = {
    'hindi': 'hi',
    'sanskrit': 'sa',
    'bengali': 'bn',
    'tamil': 'ta',
    'telugu': 'te',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'odia': 'or',
    'punjabi': 'pa',
    'urdu': 'ur',
}

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
    aud['path'] = js_data['audio_filepath'].replace('/nlsasfs/home/ai4bharat/ai4bharat-pr/speechteam/asr_datasets', '/workspace/ai4bharat-pr/speechteam/ai4bp_upload/vistaar')
    y, sr = sf.read(aud['path'])
    aud['duration'] = js_data['duration']
    aud['array'] = y
    aud['sampling_rate'] = sr
    
    return (aud, js_data['text'])
    

def main(args):
    
    with open(args.manifest_path, 'r') as f:
        data = f.read()
        splits = data.split('\n')
        if splits[-1] == '':
            splits = splits[:-1]
    
    da = Parallel(n_jobs=128)(delayed(get_data)(split) for split in tqdm(splits))
    
    dataset = eval_dataset()
    for d in da:
        dataset.fill_data(d[0], d[1])
 
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_path, device=args.device,
    )
    
    # Special case to handle odia since odia is not supported by whisper model
    if args.lang_code == 'or':
        whisper_asr.model.config.forced_decoder_ids = (
            whisper_asr.tokenizer.get_decoder_prompt_ids(
                language=None, task="transcribe"
            )
        )
    else:
        whisper_asr.model.config.forced_decoder_ids = (
            whisper_asr.tokenizer.get_decoder_prompt_ids(
                language=args.lang_code, task="transcribe"
            )
        )

    hypothesis = []
    ground_truth = []
    
    os.makedirs(dir_path + '/' + 'predictions', exist_ok=True)
    
    out_name = args.model_path.rsplit('/',1)[-1] + '_' + args.manifest_name + '_' + 'predictions.json'
    
    open(dir_path + '/' + 'predictions/' + out_name, 'w').close()
    
    st = time.time()
    
    for out in tqdm(whisper_asr(dataset, batch_size=args.batch_size), total=len(dataset)):
        
        hyp = out['text']
        ref = out['reference'][0]
        hyp = hyp.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
        ref = ref.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
        if args.lang_code[:2] != 'ur':
            hyp = normalize_sentence(hyp, args.lang_code[:2])
            ref = normalize_sentence(ref, args.lang_code[:2])
        hyp = re.sub(' +', ' ', hyp)
        ref = re.sub(' +', ' ', ref)
        
        if ref == '':
            ref = '<empty>'
        hypothesis.append(hyp)
        ground_truth.append(ref)
        res = {
            "audio_filepath":out['path'][0],
            "duration":out['duration'][0],
            "text":ref,
            "pred_text":hyp
        }
        with open(dir_path + '/' + 'predictions/' + out_name, 'a') as f:
            json.dump(res, f)
            f.write('\n')
    
    et = time.time()
     
    data = {}
    data['model'] = args.model_path
    data['dataset'] = args.manifest_name
    data['language'] = args.lang_code
    data['cer'] = jiwer.cer(ground_truth, hypothesis)
    data['time'] = (et-st)/60
    data['batch_size'] = args.batch_size
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    data['wer'] = measures['wer']

    print(data)
    
    with open(dir_path + '/' + 'results.csv', 'a') as results_fp:
        print(','.join([str(v) for v in data.values()]), file=results_fp)


if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to model",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="path to vistaar manifest",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        required=True,
        help="manifest name",
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
        default=32,
        help="Number of samples for each batch.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="current language",
    )
    args = parser.parse_args()
    
    if len(args.language) == 2:
        args.lang_code = args.language.lower()
    else:
        args.lang_code = lang_to_code[args.language.lower()]

    main(args)
