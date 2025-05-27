import regex
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()

def remove_non_characters(text):
    # Keep only Unicode letters, numbers, whitespace, and common punctuation
    cleaned_text = regex.sub(r"[^\p{L}\p{N}\p{P}\p{Z}]", "", text)
    return cleaned_text

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='left', padding_side='left')
    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def arc(model2, tokenizer2,downstream_dataset, save_file):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset['test']['question'])), desc="Processing Answers"):
        prompt = downstream_dataset['test']['question'][i]

        po_qs_input = po_prompt_ins.replace("S_P", prompt)
        po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
        po_opt_prompt = po_opt_prompt.replace("Golden Prompt:", "").strip().lstrip('\n')
        print(f"\n--- Optimized Prompt by PO Model ---\n{po_opt_prompt}")

        po_opt.append({
            'raw_question': prompt,
            "question": po_opt_prompt,
            "choices": downstream_dataset['test']['choices'][i],
            "answer": downstream_dataset['test']['answerKey'][i],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)



def BBH(model2, tokenizer2,downstream_dataset, save_file):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset)), desc="Processing Answers"):
        prompt = downstream_dataset[i]['question']

        # Optimize prompt with PO model
        po_qs_input = po_prompt_ins.replace("S_P", prompt)
        po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
        po_opt_prompt = po_opt_prompt.replace("Golden Prompt:", "").strip().lstrip('\n')
        print(f"\n--- Optimized Prompt by PO Model ---\n{po_opt_prompt}")

        po_opt.append({
            'raw_question': prompt,
            "question": po_opt_prompt,
            "choices": downstream_dataset[i]['choices'],
            "target": downstream_dataset[i]['target'],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def BBH_math(model2, tokenizer2,downstream_dataset, save_file):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset)), desc="Processing Answers"):
        prompt = downstream_dataset[i]['question']

        # Optimize prompt with PO model
        po_qs_input = po_prompt_ins.replace("S_P", prompt)
        po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
        po_opt_prompt = po_opt_prompt.replace("Golden Prompt:", "").strip().lstrip('\n')
        print(f"\n--- Optimized Prompt by PO Model ---\n{po_opt_prompt}")

        po_opt.append({
            'raw_question': prompt,
            "question": po_opt_prompt,
            "target": downstream_dataset[i]['target'],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def BBH_wordsorting(model2, tokenizer2,downstream_dataset,save_file):
    po_opt = []

    prompt = downstream_dataset[0]['question']

    # Optimize prompt with PO model
    po_qs_input = po_prompt_ins.replace("S_P", prompt)
    po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
    po_opt_prompt = po_opt_prompt.replace("Golden Prompt:", "").strip().lstrip('\n')

    for i in tqdm(range(len(downstream_dataset)), desc="Processing Answers"):
        prompt = downstream_dataset[i]['question']


        po_opt.append({
            'raw_question': prompt,
            "question": po_opt_prompt,
            "word_list": downstream_dataset[i]['word_list'],
            "target": downstream_dataset[i]['target'],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def gsm8k(model2, tokenizer2,downstream_dataset,save_file):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset['test']['question'])), desc="Processing Answers"):
        prompt = downstream_dataset['test']['question'][i]

        # Optimize prompt with PO model
        po_qs_input = po_prompt_ins.replace("S_P", prompt)
        po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
        po_opt_prompt = po_opt_prompt.replace("Golden Prompt:", "").strip().lstrip('\n')
        print(f"\n--- Optimized Prompt by PO Model ---\n{po_opt_prompt}")

        po_opt.append({
            'raw_question': prompt,
            "opt_question": po_opt_prompt,
            "answer": downstream_dataset['test']['answer'][i],
        })
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def piqa(model2, tokenizer2,downstream_dataset,  save_file):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset)), desc="Processing Answers"):
        prompt = downstream_dataset[i]['goal']

        # Optimize prompt with PO model
        po_qs_input = po_prompt_ins.replace("S_P", prompt)
        po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
        po_opt_prompt = po_opt_prompt.replace("Golden Prompt:", "").strip().lstrip('\n')
        po_opt_prompt=remove_non_characters(po_opt_prompt)
        print(f"\n--- Optimized Prompt by PO Model ---\n{po_opt_prompt}")

        po_opt.append({
            'raw_question': prompt,   "question": po_opt_prompt,
            'sol1': downstream_dataset[i]['sol1'],
            'sol2': downstream_dataset[i]['sol2'],
            'label':  downstream_dataset[i]['label'],
            'goal':  downstream_dataset[i]['goal']
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def read_json(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data


with open("./data/optimize_prompt_instruction.txt", "r",
          encoding="utf-8") as file:
    po_prompt_ins = file.read()


print("Loading PO model...")

po_model_path = "zixiaozhu/MePO" #load your model or from huggingface

model2, tokenizer2 = load_model_and_tokenizer(po_model_path)

folder='./'

for task in ['arc_easy']:#'BBH','arc_challenge','gsm8k','arc_easy''piqa'

    save_file = f"{folder}/{task}_{option}opt.json"
    if task.lower()=='arc_easy':
        downstream_dataset=load_dataset('allenai/ai2_arc', 'ARC-Easy')
        arc(model2, tokenizer2,downstream_dataset,  save_file)
    elif task.lower() == 'arc_challenge':
        downstream_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        arc(model2, tokenizer2,downstream_dataset,  save_file)
    elif task.lower()=='gsm8k':
        downstream_dataset = load_dataset("gsm8k", "main")
        gsm8k(model2, tokenizer2,downstream_dataset,  save_file)
    elif task.lower() == 'piqa':
        downstream_dataset = read_json( f'./dataset/{task}.json')
        piqa(model2, tokenizer2,downstream_dataset,  save_file)
    elif task.upper() == 'BBH':
        multiple_choice = [
            'date_understanding', 'disambiguation_qa', 'hyperbaton', 'logical_deduction_five_objects',
                           'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                           'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                           'ruin_names', 'salient_translation_error_detection', 'snarks',
                           'temporal_sequences', 'tracking_shuffled_objects_five_objects',
                           'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
                           'causal_judgement','formal_fallacies','navigate','web_of_lies',
            'sports_understanding','boolean_expressions',
                           'multistep_arithmetic_two', 'object_counting', 'word_sorting'
                           ]
        for subtask in multiple_choice:
            save_file = f"{folder}/BBH/{subtask}_{option}opt.json"
            downstream_dataset =read_json(f'./dataset/BBH/{subtask}.json')
            if subtask == 'word_sorting':
                BBH_wordsorting(model2, tokenizer2,downstream_dataset,  save_file)
            elif subtask in ['multistep_arithmetic_two', 'object_counting']:
                BBH_math(model2, tokenizer2,downstream_dataset,  save_file)
            else:
                BBH(model2, tokenizer2,downstream_dataset,  save_file)



