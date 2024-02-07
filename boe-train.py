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
import torch.nn as nn
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
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=16, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"})
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
    float_16 = True
    batch = 16
    hub_model_id_str = script_args.model_name.split("/")[1] + "_4bit_" + str(len(script_args.target_modules))
elif script_args.num_bit == 8:
    bit_4 = False
    bit_8 = True
    float_16 = True
    batch = 8
    hub_model_id_str = script_args.model_name.split("/")[1] + "_8bit_" + str(len(script_args.target_modules))
elif script_args.num_bit == 16:
    bit_4 = False
    bit_8 = False
    float_16 = True
    batch = 4
    hub_model_id_str = script_args.model_name.split("/")[1] + "_16bit_" + str(len(script_args.target_modules))
elif script_args.num_bit == 32:
    bit_4 = False
    bit_8 = False
    float_16 = False
    batch = 4
    hub_model_id_str = script_args.model_name.split("/")[1] + "_32bit_" + str(len(script_args.target_modules))

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_4bit = bit_4,
    load_in_8bit = bit_8,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token


for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model)


from peft import LoraConfig, get_peft_model
if script_args.num_bit == 16 or script_args.num_bit == 32:

  config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.1,
      bias="none",
      task_type="CAUSAL_LM"
  )
else:
  config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=script_args.target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, config)

# Step 2: Load the dataset
dataset = load_dataset(script_args.dataset_name)
dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
train_dataset = dataset['train'] 
test_dataset = dataset['test'] 
dev_dataset = dataset['validation']

print(train_dataset[0])

# Step 3: Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=batch,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=hub_model_id_str,
    fp16=float_16,
    gradient_checkpointing=script_args.gradient_checkpointing,
    output_dir="output/"+hub_model_id_str,
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)

# Step 5: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)




