import os
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
from datasets import load_dataset
from accelerate import Accelerator
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    TrainerCallback,
AutoConfig, AutoTokenizer, AutoModelForCausalLM
)

from peft import (
    get_peft_model,
    LoraConfig
)
from peft.tuners.lora import LoraLayer
from MePO_dpo_trainer_nodp import DPOTrainer
from flash_attn_patch import replace_llama_attn_with_flash_attn
import numpy as np
import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Lora attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "The dropout probability for Lora layers."}
    )
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "The loss type, either dpo or ipo."})
    if_lora: Optional[int] = field(default=1, metadata={"help": "Whether run lora or full training."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "beta in DPO/IPO loss"})

@dataclass
class DataTrainingArguments:
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
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
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )
    train_data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "The input evaluation data file (a jsonlines)."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

        with open(self.log_file, "w") as f:
            header = [
                "epoch",
                "loss",
                "rewards/chosen",
                "rewards/rejected",
                "rewards/accuracies",
                "rewards/margins",
                "logps/chosen",
                "logps/rejected",
                "logits/chosen",
                "logits/rejected",
            ]
            f.write(",".join(header) + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch

        keys_to_log = [
            "loss",
            "rewards/chosen",
            "rewards/rejected",
            "rewards/accuracies",
            "rewards/margins",
            "logps/chosen",
            "logps/rejected",
            "logits/chosen",
            "logits/rejected",
        ]
        metrics_to_log = {}

        if state.log_history:
            for entry in reversed(state.log_history):
                for key in keys_to_log:
                    if key in entry and key not in metrics_to_log:
                        metrics_to_log[key] = entry[key]

                if all(key in metrics_to_log for key in keys_to_log):
                    break


        with open(self.log_file, "a") as f:
            row = [str(epoch)]
            for key in keys_to_log:
                row.append(str(metrics_to_log.get(key, "")))
            f.write(",".join(row) + "\n")
        return control

class SaveModelCallback(TrainerCallback):

    def __init__(self, trainer, save_directory):
        self.trainer = trainer
        self.save_directory = save_directory

    def on_epoch_end(self, args, state, control, **kwargs):

        epoch = state.epoch  # 当前 epoch
        save_path = f"{self.save_directory}/checkpoint-epoch-{int(epoch)}"
        print(f"🔹 Saving model at: {save_path}")

        self.trainer.save_model(save_path)  
        self.trainer.state.save_to_json(f"{save_path}/trainer_state.json") 

        print(f"✅ Model saved at {save_path}")

def main(args=None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if args is None:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    replace_llama_attn_with_flash_attn()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    def preprocess_function(examples):
        prepared_inputs = {"prompt": [], "chosen": [], "rejected": []}
        for p, g, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            prepared_inputs["prompt"].append(p)
            prepared_inputs["chosen"].append(g)
            prepared_inputs["rejected"].append(r)
        return prepared_inputs
    
    label_ignore_id = -100

    print("start data preprocess")
    data_files = {}
    data_files["train"] = data_args.train_data_path
    raw_datasets = load_dataset(
        "json",
        data_files=data_files
    )
    column_names = raw_datasets["train"].column_names
    prepared_dataset = raw_datasets.map(
        preprocess_function,
        batched=True,
        batch_size=len(raw_datasets["train"]),
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Running tokenizer on train dataset"
    )
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(prepared_dataset["train"]), data_args.max_train_samples)
        prepared_dataset["train"] = prepared_dataset["train"].select(range(max_train_samples))
    logger.info("load dataset finished")


    accelerator = Accelerator()
    # load config and tokenziers
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.use_cache = False



    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, load_in_8bit=True,  device_map="auto")



    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left', padding_side='left')


    if model_args.if_lora == 0: #1
        ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, load_in_8bit=True,  device_map="auto")

    if hasattr(model, "enable_input_require_grads"): #true
        model.enable_input_require_grads()

    if hasattr(model, "gradient_checkpointing_enable"):#true
        model.gradient_checkpointing_enable()


    model = accelerator.prepare(model)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup seed
    set_seed(training_args.seed)
    if len(tokenizer) > tokenizer.vocab_size: #false
        model.resize_token_embeddings(len(tokenizer))
        if model_args.if_lora == 0:
            ref_model.resize_token_embeddings(len(tokenizer))

    # Setup Trainer
    training_args = training_args.to_dict()
    training_args.update({'remove_unused_columns': False})
    training_args = TrainingArguments(**training_args)
    peft_config = LoraConfig(
        r=model_args.lora_r,#8
        lora_alpha=model_args.lora_alpha, #16
        lora_dropout=model_args.lora_dropout, #0.05
        target_modules=[
            "q_proj",
            "v_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if model_args.if_lora != 0:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        ref_model = None
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=model_args.beta, # DPO temprature 0.01
        train_dataset=prepared_dataset["train"],
        tokenizer=tokenizer,
        args=training_args,
        max_length=data_args.model_max_length,
        max_prompt_length=int(data_args.model_max_length) * 3 // 4,
        loss_type=model_args.loss_type,

    )
    trainer.add_callback(SaveModelCallback(trainer, save_directory=f"{training_args.output_dir}/saved_models"))
    loss_log_file = f"{training_args.output_dir}/epoch_loss.csv"
    trainer.add_callback(LossLoggerCallback(loss_log_file))
    # Training
    print(f"Before training: {type(trainer.model)}")
    train_result = trainer.train()
    print(f"After training: {type(trainer.model)}")  

    trainer.save_state()
    

    trainer.save_model()
