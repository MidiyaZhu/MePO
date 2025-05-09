
ROOTPATH = "YOUR_PATH"  # 修改为你的实际路径

import PattoPO_train_dpo


# 训练参数
MAXLEN = 2048
EPOCH =3
ISLORA = 1
SETTING = "sigmoid"
BETA = 0.01

models = ["PattoPO_optimizer"]

for model in models:

    raw_model_path = f"/cache/transformers/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
    train_data_path = f"./data/PattoPo_Prompt_Preference_Dataset.jsonl"


    model_output_path = f"./{model}_{SETTING}_peft/"
    final_model_output_path = f"./{model}-{SETTING}-lora/"

    PER_GPU_BATCH =1
    GRA_ACC = 4


    # **调用 train_dpo.py**
    args = [
        "--model_name_or_path", raw_model_path,
        "--bf16", "True",
        "--output_dir", model_output_path,
        "--num_train_epochs", str(EPOCH),
        "--per_device_train_batch_size", str(PER_GPU_BATCH),
        "--gradient_accumulation_steps", str(GRA_ACC),
        "--save_strategy", "epoch",
        "--save_total_limit", "1",
        "--eval_steps", "500",
        "--learning_rate", "1e-6",
        "--log_level", "info",
        "--logging_strategy", "steps",
        "--logging_steps", "1",
        "--lr_scheduler_type", "linear",
        "--tf32", "True",
        "--model_max_length", str(MAXLEN),
        "--train_data_path", train_data_path,
        "--preprocessing_num_workers", "32",
        "--dataloader_num_workers", "32",
        "--gradient_checkpointing", "True",
        "--report_to", "none",
        "--if_lora", str(ISLORA),
        "--beta", str(BETA),
        "--loss_type", SETTING
    ]

    PattoPO_train_dpo.main(args)


    from PattoPO_merge_peft_adapter import main as merge_main
    merge_args = [
        "--adapter_model_name", model_output_path,
        "--base_model_name", raw_model_path,
        "--output_name", final_model_output_path
    ]
    merge_main(merge_args)
