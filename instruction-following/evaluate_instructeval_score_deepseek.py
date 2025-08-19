import json
import openai
from tqdm import tqdm
import re
import os
from openai import OpenAI
openai.api_key = "YOUR-DS_KEY"

def read_json(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data
def selfeval(prompt_ins,data, save_file):
    if os.path.exists(save_file):
        result = read_json(save_file)
        for i in tqdm(range(len(result), len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            opt_answer = data[i]['opt_ans']
            input = data[i]['input']


            opt_question = f'{opt_prompt}\n{input}'

            prompt = prompt_ins.replace("{INS}", opt_question).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            score = opt_ans.choices[0].message.content

            result.append({
                'instruction': data[i]['instruction'],
                'opt_ans':  data[i]['opt_ans'],
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": data[i]['opt_question'],
                "score": score,

            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
    else:
        result=[]
        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            opt_answer = data[i]['opt_ans']
            input = data[i]['input']

            opt_question = f'{opt_prompt}\n{input}'

            prompt = prompt_ins.replace("{INS}", opt_question).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )

            score = opt_ans.choices[0].message.content

            result.append({
                'instruction': data[i]['instruction'],
                'opt_ans': data[i]['opt_ans'],
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": data[i]['opt_question'],
                "score": score,

            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
def selfeval_raw(prompt_ins,data, save_file):
    if os.path.exists(save_file):
        result = read_json(save_file)
        for i in tqdm(range(len(result), len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['instruction']
            opt_answer = data[i]['opt_ans']
            input = data[i]['input']

            opt_question = f'{opt_prompt}\n{input}'

            prompt = prompt_ins.replace("{INS}", opt_question).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            score = opt_ans.choices[0].message.content

            result.append({
                'instruction': data[i]['instruction'],
                'opt_ans': data[i]['opt_ans'],
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": data[i]['opt_question'],
                "score": score,

            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
    else:
        result=[]
        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['instruction']
            opt_answer = data[i]['opt_ans']
            input = data[i]['input']

            opt_question = f'{opt_prompt}\n{input}'

            prompt = prompt_ins.replace("{INS}", opt_question).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            score = opt_ans.choices[0].message.content

            result.append({
                'instruction': data[i]['instruction'],
                'opt_ans': data[i]['opt_ans'],
                "input": data[i]['input'],
                "output": data[i]['output'],
                "opt_question": data[i]['opt_question'],
                "score": score,

            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
def vicuna(prompt_ins,data, save_file):
    if os.path.exists(save_file):
        result = read_json(save_file)
        for i in tqdm(range(len(result), len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            opt_answer = data[i]['opt_ans']
            prompt = prompt_ins.replace("{INS}", opt_prompt).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            score = opt_ans.choices[0].message.content

            result.append({
                'instruction': data[i]['instruction'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": opt_answer,
                "response": data[i]['response'],
                "score":score,

            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
    else:
        result = []
        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            opt_answer = data[i]['opt_ans']
            prompt = prompt_ins.replace("{INS}", opt_prompt).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            score = opt_ans.choices[0].message.content

            result.append({
                'instruction': data[i]['instruction'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": opt_answer,
                "response": data[i]['response'],
                "score": score,

            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
def BPO_test(prompt_ins,data, save_file):
    if os.path.exists(save_file):
        result = read_json(save_file)
        for i in tqdm(range(len(result), len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            opt_answer = data[i]['opt_ans']
            prompt= prompt_ins.replace("{INS}", opt_prompt).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            score = opt_ans.choices[0].message.content

            result.append({
                'prompt': data[i]['prompt'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": opt_answer,
                "bpo_opt_prompt": data[i]['bpo_opt_prompt'],
                "bpo_good_res": data[i]['bpo_good_res'],
                "bpo_bad_res": data[i]['bpo_bad_res'],
                "score": score,
            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
    else:
        result=[]
        for i in tqdm(range(len(data)), desc="Processing Answers"):
            opt_prompt = data[i]['opt_question']
            opt_answer = data[i]['opt_ans']
            prompt = prompt_ins.replace("{INS}", opt_prompt).replace("{RES}", opt_answer)
            client = OpenAI(api_key=openai.api_key, base_url="https://api.deepseek.com")
            opt_ans = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            score = opt_ans.choices[0].message.content

            result.append({
                'prompt': data[i]['prompt'],
                "opt_question": data[i]['opt_question'],
                "opt_ans": opt_answer,
                "bpo_opt_prompt": data[i]['bpo_opt_prompt'],
                "bpo_good_res": data[i]['bpo_good_res'],
                "bpo_bad_res": data[i]['bpo_bad_res'],
                "score": score,
            })
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)


def read_txt(filename):
    """Reads a text file and returns its content as a string."""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()



folder='Your_data_path'
score_tmp='instruction_eval_score.txt'
prompt_ins = read_txt(score_tmp)
for llm in ['qwen25_7b']:#'vicuna','qwen2_7b','deepseek',
    for task in [ 'selfeval', 'BPO_test', 'vicuna']:
        data_file = f"{folder}/{llm}/{task}_raw_ans.json"
        save_file = f"{folder}/{llm}/{task}_raw_ans_score.json"
        data = read_json(data_file)

        if task == 'selfeval':
            selfeval_raw(prompt_ins, data, save_file)
        elif task == 'BPO_test':
            BPO_test(prompt_ins, data, save_file)
        elif task == 'vicuna':
            vicuna(prompt_ins, data, save_file)

        for option in ['po',]:  # 'base','tmp''po'
            data_file = f"{folder}/{llm}/{task}_{option}opt_ans.json"
            data = read_json(data_file)

            save_file = f"{folder}/{llm}/{task}_{option}opt_ans_score.json"

            if task == 'selfeval':
                selfeval(prompt_ins, data, save_file)

            elif task == 'BPO_test':
                BPO_test(prompt_ins, data, save_file)
            elif task == 'vicuna':
                vicuna(prompt_ins, data, save_file)
#
