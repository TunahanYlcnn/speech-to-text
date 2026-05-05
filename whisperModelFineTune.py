import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, WhisperProcessor, EarlyStoppingCallback
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

df = pd.read_excel("/content/drive/MyDrive/kayit4.xlsx")
dataset = Dataset.from_pandas(df)
common_voice = DatasetDict()
train_test_split = dataset.train_test_split(test_size=0.2)
common_voice["train"] = train_test_split["train"]
common_voice["test"] = train_test_split["test"]

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language="Turkish", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="Turkish", task="transcribe")
common_voice = common_voice.cast_column("ses", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["ses"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["metin"], max_length = 448, truncation=True).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.generation_config.update(language="Turkish", task="transcribe", forced_decoder_ids=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-hi",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=9,
    warmup_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
processor.save_pretrained(training_args.output_dir)

trainer.train()