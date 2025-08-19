
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import re

def extract_answer(text):
    """Extracts the answer after '####' in the given text."""
    match = re.search(r'####\s*(\d+)', text)
    return match.group(1) if match else None

def generate_response(model, tokenizer, prompt):
    """Generates a response from the given model and tokenizer based on the provided prompt."""
    messages = [
        {"role": "system",
         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
def read_txt(filename):
    """Reads a text file and returns its content as a string."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()

def read_json(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data
def load_model_and_tokenizer(model_path):
    """Loads a model and tokenizer from the given path."""
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",load_in_4bit=True,resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='left', padding_side='left')
    return model, tokenizer



def selfeval(model, tokenizer,data,output_file):

    if os.path.exists(output_file):
        results = read_json(output_file)
        for i in tqdm(range(len(results),len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            input = data[i]['input']

            raw_prompt = data[i]['instruction']

            opt_question = f'Instruction: {opt_prompt}\n\nInput: \'{input}\'\n\nOutput: '

            raw_response = generate_response(model, tokenizer, opt_question)

            results.append({
                'instruction': raw_prompt,
                'opt_ans': raw_response,
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": opt_prompt,
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_file}")
    else:

        results = []

        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            input = data[i]['input']

            raw_prompt = data[i]['instruction']

            opt_question = f'Instruction: {opt_prompt}\n\nInput: \'{input}\'\n\nOutput: '

            raw_response = generate_response(model, tokenizer, opt_question)

            results.append({
                'instruction': raw_prompt,
                'opt_ans': raw_response,
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": opt_prompt,
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")


def selfeval_raw(model, tokenizer, data, output_file):
    if os.path.exists(output_file):
        results = read_json(output_file)
        for i in tqdm(range(len(results), len(data)), desc="Processing Answers"):

            input = data[i]['input']

            raw_prompt = data[i]['instruction']

            raw_question = f'Instruction: {raw_prompt}\n\nInput: \'{input}\'\n\nOutput: '

            raw_response = generate_response(model, tokenizer, raw_question)

            results.append({
                'instruction': raw_prompt,
                'opt_ans': raw_response,
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": data[i]['opt_question'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_file}")
    else:

        results = []

        for i in tqdm(range(len(data)), desc="Processing Answers"):
            input = data[i]['input']

            raw_prompt = data[i]['instruction']

            raw_question = f'Instruction: {raw_prompt}\n\nInput: \'{input}\'\n\nOutput: '

            raw_response = generate_response(model, tokenizer, raw_question)

            results.append({
                'instruction': raw_prompt,
                'opt_ans': raw_response,
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": data[i]['opt_question'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

def BPO_test_raw(model, tokenizer, data, output_file):
    if os.path.exists(output_file):
        results = read_json(output_file)
        for i in tqdm(range(len(results), len(data)), desc="Processing Answers"):

            # opt_prompt = data[i]['opt_question']
            raw_prompt = data[i]['bpo_opt_prompt']


            raw_response = generate_response(model, tokenizer, raw_prompt)

            results.append({
                'prompt': data[i]['prompt'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "bpo_opt_prompt": data[i]['bpo_opt_prompt'],
                "bpo_good_res": data[i]['bpo_good_res'],
                "bpo_bad_res": data[i]['bpo_bad_res'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_file}")
    else:

        results = []

        for i in tqdm(range(len(data)), desc="Processing Answers"):
            # opt_prompt = data[i]['opt_question']
            raw_prompt = data[i]['bpo_opt_prompt']

            raw_response = generate_response(model, tokenizer, raw_prompt)

            results.append({
                'prompt': data[i]['prompt'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "bpo_opt_prompt": data[i]['bpo_opt_prompt'],
                "bpo_good_res": data[i]['bpo_good_res'],
                "bpo_bad_res": data[i]['bpo_bad_res'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

def BPO_test(model, tokenizer, data, output_file):
    if os.path.exists(output_file):
        results = read_json(output_file)
        for i in tqdm(range(len(results), len(data)), desc="Processing Answers"):

            opt_prompt = data[i]['opt_question']
            # raw_prompt = data[i]['bpo_opt_prompt']


            raw_response = generate_response(model, tokenizer, opt_prompt)

            results.append({
                'prompt': data[i]['prompt'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "bpo_opt_prompt": data[i]['bpo_opt_prompt'],
                "bpo_good_res": data[i]['bpo_good_res'],
                "bpo_bad_res": data[i]['bpo_bad_res'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_file}")
    else:

        results = []

        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            # raw_prompt = data[i]['bpo_opt_prompt']

            raw_response = generate_response(model, tokenizer, opt_prompt)

            results.append({
                'prompt': data[i]['prompt'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "bpo_opt_prompt": data[i]['bpo_opt_prompt'],
                "bpo_good_res": data[i]['bpo_good_res'],
                "bpo_bad_res": data[i]['bpo_bad_res'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

def vicuna_raw(model, tokenizer, data, output_file):
    if os.path.exists(output_file):
        results = read_json(output_file)
        for i in tqdm(range(len(results), len(data)), desc="Processing Answers"):

            # opt_prompt = data[i]['opt_question']
            raw_prompt = data[i]['instruction']


            raw_response = generate_response(model, tokenizer, raw_prompt)

            results.append({
                'instruction': data[i]['instruction'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "response": data[i]['response'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_file}")
    else:

        results = []

        for i in tqdm(range(len(data)), desc="Processing Answers"):
            # opt_prompt = data[i]['opt_question']
            raw_prompt = data[i]['instruction']

            raw_response = generate_response(model, tokenizer, raw_prompt)

            results.append({
                'instruction': data[i]['instruction'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "response": data[i]['response'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
def vicuna(model, tokenizer, data, output_file):
    if os.path.exists(output_file):
        results = read_json(output_file)
        for i in tqdm(range(len(results), len(data)), desc="Processing Answers"):

            opt_prompt = data[i]['opt_question']
            # raw_prompt = data[i]['bpo_opt_prompt']


            raw_response = generate_response(model, tokenizer, opt_prompt)

            results.append({
                'instruction': data[i]['instruction'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "response": data[i]['response'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {output_file}")
    else:

        results = []

        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            # raw_prompt = data[i]['bpo_opt_prompt']

            raw_response = generate_response(model, tokenizer, opt_prompt)

            results.append({
                'instruction': data[i]['instruction'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": raw_response,
                "response": data[i]['response'],
            })

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':


    # Load first model and tokenizer
    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

    model, tokenizer = load_model_and_tokenizer(model_name_or_path)


    folder = 'your_data_path'
    for task in [ 'selfeval', 'BPO_test', 'vicuna']:
        data_file = f"{folder}/{task}_poopt.json"
        data = read_json(data_file)
        save_file = f"{folder}/qwen25_7b/{task}_raw_ans.json"

        '''raw dataset'''
        if task == 'selfeval':
            selfeval_raw(model, tokenizer, data, save_file)
        elif task == 'BPO_test':
            BPO_test_raw(model, tokenizer, data, save_file)
        elif task == 'vicuna':
            vicuna_raw(model, tokenizer, data, save_file)
        '''raw dataset'''

        '''optimized dataset'''
        for option in ['po']:  # 'base','tmp''po'
            data_file = f"{folder}/{task}_{option}opt.json"
            data = read_json(data_file)

            save_file = f"{folder}/qwen25_7b/{task}_{option}opt_ans.json"

            if task == 'selfeval':
                selfeval(model, tokenizer,data, save_file)

            elif task == 'BPO_test':
                BPO_test(model, tokenizer, data, save_file)
            elif task == 'vicuna':
                vicuna(model, tokenizer, data, save_file)
        '''raw dataset'''