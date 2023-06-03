import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from datasets import DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset, load_from_disk, Audio
from torch.utils.data import IterableDataset
import pandas as pd
from datasets import Dataset
import pyarrow as pa

import evaluate
import transformers
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
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import librosa
import soundfile as sf
import datasets
from pathlib import Path

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.18.2", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    model_index_name: str = field(default=None, metadata={"help": "Pretty name for the model card."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    do_remove_punctuation: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be striped of punctuation."},
    )
    do_normalize_eval: bool = field(
        default=True,
        metadata={"help": "Whether to normalise the references and predictions in the eval WER calculation."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "The number of streamed examples to download before shuffling them. The large the buffer, "
                "the closer it is to real offline shuffling."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use streaming mode to load and pre-process the data."},
    )


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: torch.from_numpy(self.processor.feature_extractor(sample['audio']["array"], sampling_rate=sample['audio']["sampling_rate"])['input_features'][0])} for sample in features]
        label_features = [{"input_ids": torch.tensor(self.processor.tokenizer(sample['sentence'])['input_ids'])} for sample in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def get_data(split):
    js_data = json.loads(split)
    aud = {}
    
    aud['path'] = js_data['audio_filepath'].replace('/nlsasfs/home/ai4bharat/', '/workspace/')
    aud['text'] = js_data['text']

    return (aud['path'], aud['text'])

def load_maybe_streaming_dataset(dataset_name, dataset_config_name, split="train", streaming=True, **kwargs):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    
    with open(dataset_name + '/' + split + '.json', 'r') as f:
        data = f.read()
        splits = data.split('\n')[:-1]
    
    da = Parallel(n_jobs=128)(delayed(get_data)(split) for split in tqdm(splits))
    data_dict = {}
    data_dict['path'] = []
    data_dict['sentence'] = []
    
    for d in da:
        data_dict['path'].append(d[0])
        data_dict['sentence'].append(d[1])
        
    dataset = Dataset.from_dict({"audio": data_dict['path'], "sentence": data_dict['sentence']}).cast_column("audio", Audio())
    return dataset


def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.local_rank = 0
    training_args.dataloader_num_workers=32
    training_args.remove_unused_columns=False

    datasets.config.DOWNLOADED_DATASETS_PATH = Path(model_args.cache_dir)
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    #send_example_telemetry("run_speech_recognition_seq2seq_streaming", model_args, data_args)
    print(data_args.streaming)
    print(data_args.max_eval_samples)
    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    #4. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )

    if training_args.do_eval:
        raw_datasets["eval"] = load_maybe_streaming_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})
    
    ##adding droput
    #config.update({"dropout" : 0.05, "attention_dropout" : 0.05})

    if training_args.gradient_checkpointing:
        config.update({"use_cache": False})

    MAX_DURATION_IN_SECONDS = 30.0
    max_input_length = MAX_DURATION_IN_SECONDS * 16000
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=max_input_length
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=448
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    do_remove_punctuation = data_args.do_remove_punctuation
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets

    # 8. Load Metric
    metric = evaluate.load("wer")
    do_normalize_eval = data_args.do_normalize_eval

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # 9. Create a single speech processor
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 11. Configure Trainer
    # Trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
    # Only required for streaming: Trainer automatically shuffles non-streaming datasets
    class ShuffleCallback(TrainerCallback):
        def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
            if isinstance(train_dataloader.dataset, IterableDatasetShard):
                pass  # set_epoch() is handled by the Trainer
            elif isinstance(train_dataloader.dataset, IterableDataset):
                train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

    #Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[ShuffleCallback()] if data_args.streaming else None,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        print(training_args)
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 14. Write Training Stats
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
        if "common_voice" in data_args.dataset_name:
            kwargs["language"] = data_args.dataset_config_name.split('-')[0]
        if model_args.model_index_name is not None:
            kwargs["model_name"] = model_args.model_index_name

    return results


if __name__ == "__main__":
    main()
