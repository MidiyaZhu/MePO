import random
from transformers import AutoTokenizer
import json
import re

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

degrade_flag=True


def build_multipledataset(degrade_flag, datasetname):
    if degrade_flag == True:
        save_path = "./data/POP_Dataset_degrade.jsonl"
    else:
        save_path = "./data/POP_Dataset.jsonl"
    w_dpo_list = []
    count = 0
    for dd in datasetname:

        with open( f"./data/deepseek_evaluate_{dd}.json",
                "r", encoding="utf-8") as f:
            data = json.load(f)

        newdict = {}
        for index in data:
            raw_prompt = index['raw_prompt']
            newdict[raw_prompt] = {'score': [], 'acc': [], 'new_prompt': [], 'response': [], 'type': []}

        for index in data:
            raw_prompt = index['raw_prompt']
            if index['score'] == None and index['accuracy'] != None:
                index['score'] = index['accuracy']
            if index['accuracy'] == None and index['score'] != None:
                index['accuracy'] = index['score']

            newdict[raw_prompt]['score'].append(index['score'])
            newdict[raw_prompt]['acc'].append(index['accuracy'])
            newdict[raw_prompt]['new_prompt'].append(index['new_prompt'])
            newdict[raw_prompt]['response'].append(index['response'])
            if 'type' in index:
                newdict[raw_prompt]['type'].append(index['type'])
            else:
                print('error')


        keys_to_delete = []

        for key in newdict:
            if newdict[key]['type'] != ['raw', 'optimized']:
                newtype = newdict[key]['type']
                print(f'Swap after {newtype}\n')
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del newdict[key]

        random.seed(2024)

        prompts = json.load(
            open("./data/qwen_prompts.json", "r"))
        basic_prompt = prompts["optimizer_qwen"]


        model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation_side='left', padding_side='left')

        for prompt in newdict:
            score = newdict[prompt]['score']
            if len(score)==2:
                if score[0] and score[1]:
                    if score[1] > 8 or score[1] == 8:
                        if score[1] > score[0] or score[1] == score[0]:

                            chosen = newdict[prompt]['new_prompt'][1]
                            rejected = newdict[prompt]['new_prompt'][0]
                            sliver_response = newdict[prompt]['response'][0]
                            golden_response = newdict[prompt]['response'][1]
                            raw_prompt = prompt
                            if contains_chinese(chosen) or contains_chinese(rejected) or contains_chinese(
                                    golden_response) or contains_chinese(sliver_response):
                                continue
                            new_prompt_temp = basic_prompt.replace("S_P", rejected) \
                                .replace("S_R", sliver_response) \
                                .replace("G_R", golden_response)

                            max_length = 2000
                            while True:
                                messages = [
                                    {"role": "system",
                                     "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                    {"role": "user", "content": new_prompt_temp}
                                ]
                                final_prompt = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )
                                #
                                prompt_tokens = tokenizer.encode(final_prompt)

                                if len(prompt_tokens) <= max_length:
                                    break
                                #
                                if len(sliver_response) > 10:
                                    sliver_response = sliver_response[:-10]
                                else:
                                    break

                                new_prompt_temp = basic_prompt.replace("S_P", rejected) \
                                    .replace("S_R", sliver_response) \
                                    .replace("G_R", golden_response)

                            #
                            messages = [
                                {"role": "system",
                                 "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                                {"role": "user", "content": new_prompt_temp}
                            ]
                            new_prompt = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )

                            w_dpo_list.append({
                                "id": count,
                                'chosen': chosen,
                                'rejected': rejected,
                                'sliver_response': sliver_response,
                                'golden_response': golden_response,
                                'raw_prompt': raw_prompt,
                                'prompt': new_prompt
                            })
                            count += 1
        print(f'{dd} count: {count}\n')
        if degrade_flag == True:
            with open(
                    f"./data/degrade/qwen_degrade_{dd}_ans.json",
                    "r", encoding="utf-8") as f:
                degradedata = json.load(f)

            for degrade in degradedata:
                raw_prompt = degrade['raw_prompt']
                if raw_prompt in newdict:
                    rejected = degrade['degrade_prompt']
                    sliver_response = degrade['degrade_response']
                    chosen = newdict[raw_prompt]['new_prompt'][1]
                    golden_response = newdict[raw_prompt]['response'][1]
                    if contains_chinese(chosen) or contains_chinese(rejected) or contains_chinese(
                            golden_response) or contains_chinese(
                            sliver_response):
                        continue
                    new_prompt_temp = basic_prompt.replace("S_P", rejected) \
                        .replace("S_R", sliver_response) \
                        .replace("G_R", golden_response)

                    max_length = 2000
                    while True:
                        messages = [
                            {"role": "system",
                             "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                            {"role": "user", "content": new_prompt_temp}
                        ]
                        final_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        prompt_tokens = tokenizer.encode(final_prompt)

                        if len(prompt_tokens) <= max_length:
                            break

                        if len(sliver_response) > 10:
                            sliver_response = sliver_response[:-10]
                        else:
                            break

                        new_prompt_temp = basic_prompt.replace("S_P", rejected) \
                            .replace("S_R", sliver_response) \
                            .replace("G_R", golden_response)

                    messages = [
                        {"role": "system",
                         "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": new_prompt_temp}
                    ]
                    new_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    w_dpo_list.append({
                        "id": count,
                        'chosen': chosen,
                        'rejected': rejected,
                        'sliver_response': sliver_response,
                        'golden_response': golden_response,
                        'raw_prompt': raw_prompt,
                        'prompt': new_prompt
                    })
                    count += 1
            print(f'{dd} degraded added count: {count}\n')
    print(f'total degraded added count: {count}\n')
    random.shuffle(w_dpo_list)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in w_dpo_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


datasetname=['alpaca','BPO',]
build_multipledataset(degrade_flag,datasetname)
