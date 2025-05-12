
from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import PeftModel
from transformers import  HfArgumentParser,AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file , save_file
@dataclass
class ScriptArguments:
    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})

def translate_state_dict_key(k):
    return k.replace("base_model.model.base_model.model.model.layers", "base_model.model.model.layers")

def main(args=None):
    parser = HfArgumentParser((ScriptArguments))
    if args is None:
        script_args = parser.parse_args_into_dataclasses()[0]
    else:
        script_args = parser.parse_args_into_dataclasses(args)[0]


    # load config and tokenziers
    config = AutoConfig.from_pretrained(script_args.base_model_name)
    config.use_cache = False
    # use truncation_side='left' to preserve linking between end of prompt and target labels
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name, truncation_side='left')
    # initialize modules
    model = AutoModelForCausalLM.from_pretrained(script_args.base_model_name, config=config)

    if tokenizer.pad_token is None: #false
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = 0

    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    print("-" * 20 + "Weight before merge" + "-" * 20)
    old_weight = model.get_output_embeddings().weight.clone()
    print(old_weight)

    # check Lora weights before merge
    print("-" * 20 + "Check Lora Weights before merge" + "-" * 20)

    lora_weights = load_file(os.path.join(script_args.adapter_model_name, "adapter_model.safetensors"))

    for k, v in lora_weights.items():
        if "lora_B" in k:
            print(k)
            print(v)
            break
    new_lora_dict = {}
    for k, v in lora_weights.items():
        new_lora_dict[translate_state_dict_key(k)] = v


    save_file(new_lora_dict, os.path.join(script_args.adapter_model_name, "adapter_model.safetensors"))

    # Load the Lora model
    model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
    model.eval()

    # check Lora weights during merge
    print("-" * 20 + "Check Lora Weights during merge" + "-" * 20)
    for k, v in model.state_dict().items():
        if "lora_B" in k:
            print(k)
            print(v)
            break

    # merge lora weight and base model
    model = model.merge_and_unload()

    print("-" * 20 + "Weight after merge" + "-" * 20)
    new_weight = model.get_output_embeddings().weight.clone()
    print(new_weight)

    print("-" * 20 + "Weight difference" + "-" * 20)
    print((new_weight - old_weight).sum())

    model.half().save_pretrained(f"{script_args.output_name}")
    tokenizer.save_pretrained(f"{script_args.output_name}")
