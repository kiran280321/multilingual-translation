import torch
from datasets import load_dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
SRC_LANG="en"
TGT_LANG="de"
WMT_CONFIGS={
    ("en","de"):"de-en",
    ("de","en"):"de-en",
    ("en","fr"):"fr-en",
    ("fr","en"):"fr-en",
    ("en","hi"):"hi-en",
    ("hi","en"):"hi-en",
    ("en","ru"):"ru-en",
    ("ru","en"):"ru-en",
    ("en","cs"):"cs-en",
    ("cs","en"):"cs-en",
}
if (SRC_LANG, TGT_LANG) not in WMT_CONFIGS:
    raise ValueError(f"WMT dataset not available for {SRC_LANG}->{TGT_LANG}")
dataset_config=WMT_CONFIGS[(SRC_LANG,TGT_LANG)]
print(f"Loading WMT dataset:{dataset_config}")
dataset=load_dataset("wmt14",dataset_config)
MODEL_NAME=f"Helsinki-NLP/opus-mt-{SRC_LANG}-{TGT_LANG}"
print(f"Loading model:{MODEL_NAME}")
tokenizer=MarianTokenizer.from_pretrained(MODEL_NAME)
model=MarianMTModel.from_pretrained(MODEL_NAME)
def preprocess_function(examples):
    inputs,targets=[],[]
    for ex in examples["translation"]:
        if SRC_LANG in ex and TGT_LANG in ex:
            inputs.append(ex[SRC_LANG])
            targets.append(ex[TGT_LANG])
    model_inputs=tokenizer(
        inputs,max_length=128,truncation=True,padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels=tokenizer(
            targets,max_length=128,truncation=True,padding="max_length"
        )
    model_inputs["labels"]=labels["input_ids"]
    return model_inputs
print("Tokenizing dataset")
tokenized_datasets=dataset.map(preprocess_function,batched=True,remove_columns=dataset["train"].column_names)
batch_size=8
args=Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    do_eval=True
)
data_collator=DataCollatorForSeq2Seq(tokenizer,model=model)
trainer=Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_datasets["validation"].shuffle(seed=42).select(range(500)),
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Starting fine-tuning")
trainer.train()
save_dir=f"./fine_tuned_{SRC_LANG}_{TGT_LANG}"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Fine-tuned model saved at {save_dir}")
