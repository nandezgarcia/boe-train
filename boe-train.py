# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer

# from trl import SFTTrainer, is_xpu_available
from huggingface_hub import login
import transformers


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="bertin-project/bertin-gpt-j-6B", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="VanoInvestigations/BOE_with_BERTIN_for_tokenize_2045", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    report_to: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "Mixed precision training"})
    target_modules: Optional[str] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})
    num_bit: Optional[int] = field(default=32, metadata= {"help" : "Num bit to load"})

    


parser = HfArgumentParser(ScriptArguments)
print(parser.parse_args_into_dataclasses()[0])
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.num_bit == 4:
    bit_4 = True
    bit_8 = False
    float_16 = False
    script_args.use_peft = True
    hub_model_id_str = script_args.model_name.split("/")[1] + "_4bit_" + str(len(script_args.target_modules))
elif script_args.num_bit == 8:
    bit_4 = False
    bit_8 = True
    float_16 = True
    script_args.use_peft = True
    hub_model_id_str = script_args.model_name.split("/")[1] + "_8bit_" + str(len(script_args.target_modules))
elif script_args.num_bit == 16:
    bit_4 = False
    bit_8 = False
    float_16 = True
    hub_model_id_str = script_args.model_name.split("/")[1] + "_16bit_" + str(len(script_args.target_modules))
elif script_args.num_bit == 32:
    bit_4 = False
    bit_8 = False
    float_16 = False
    hub_model_id_str = script_args.model_name.split("/")[1] + "_16bit_" + str(len(script_args.target_modules))

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_4bit = bit_4,
    load_in_8bit = bit_8,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, truncation=True, seq_length=1021)
tokenizer.add_special_tokens({'pad_token': '<pad>'})

# Step 2: Load the dataset
dataset = load_dataset(script_args.dataset_name)
dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
train_dataset = dataset['train'] 
test_dataset = dataset['test'] 
dev_dataset = dataset['validation']

# Step 3: Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.report_to,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=hub_model_id_str,
    fp16=float_16,
    gradient_checkpointing=script_args.gradient_checkpointing,
    output_dir="output/"+hub_model_id_str,
    use_cpu=True,
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    from peft import LoraConfig, get_peft_model
    
    target_modeles_list = script_args.target_modules.split(",")
    print(type(target_modeles_list))
    print(target_modeles_list)

    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=target_modeles_list,
    )
    model = get_peft_model(model, peft_config)
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)


