import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain.schema import Document
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import json

file_path = 'path/to/your/anonymized_data.txt'
# Open the file in read mode and read the entire content
with open(file_path, 'r') as file:
    text = file.read()
  
docs = [Document(page_content=text)]
analyzed_fields = [
    "PESRON",
    "DATE",
    "PHONE_NUMBER",
    "ORGANIZATION",
    "LOCATION",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "URL",
    "IP_ADDRESS",
    "CRYPTO",
    "NRP",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
    "US_TIN",
    "US_ZIP_CODE"]

def split_text(text, max_length):
    """
    Split the text into chunks of max_length characters.
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Split the text into smaller chunks
max_chunk_length = 1000000  # Set a suitable chunk length
text_chunks = split_text(text, max_chunk_length)

# Process each chunk separately
private_chunks = []
privacy_and_adding_noise = PresidioReversibleAnonymizer(add_default_faker_operators=False, analyzed_fields=analyzed_fields)

for chunk in text_chunks:
    private_chunk = privacy_and_adding_noise.anonymize(chunk)
    private_chunks.append(private_chunk)

# Combine the anonymized chunks back into a single text
protected_text = "".join(private_chunks)

lines = anonymized_text.split('\n')
formatted_data = [{"text": line} for line in lines if line.strip()]

with open("formatted_dataset.json", "w") as f:
    json.dump(formatted_data, f)

model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset = load_dataset("json", data_files="formatted_dataset.json", split="train")
new_model = "Llama-2-7b-chat-finetune"

lora_r = 64
lora_alpha = 64
lora_dropout = 0.01
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output = "./results"
num_epochs = 1
fp16  = False
bf16  = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
max_seq_length = None
packing = False
device_map = {"":0}

comput_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=comput_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
if comput_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=output,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=10,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    packing=packing,
)
trainer.train()

trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
