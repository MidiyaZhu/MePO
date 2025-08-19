import json
import re
import os
import pandas as pd

def extract_assistant_content(text):
    match = re.search(r'<\|assistant\|>\s*"([\s\S]+?)"', text)
    return match.group(1) if match else None

def extract_sliver_prompt(text):
    match = re.search(r'Sliver Prompt:\s*(.*?)\s*```', text, re.DOTALL)
    return match.group(1).strip() if match else None

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
def extract_last_number(text):
    """Extracts the last number (including negatives) from a given text."""
    numbers = re.findall(r'-?\d+', text)
    return int(numbers[-1]) if numbers else None

def extract_lasttwo_number(text):
    """Extracts the last number (including negatives) from a given text."""
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        if len(numbers) > 1:
            return int(numbers[-2])
    return None
def extract_lasttee_number(text):
    """Extracts the last number (including negatives) from a given text."""
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        if len(numbers)>2:
            return int(numbers[-3])
    return  None

def extract_last_bold_number(text):
    matches = re.findall(r"\\(?:\(|\[)\\boxed\{(-?\d+)\}\\(?:\)|\])", text)
    return int(matches[-1]) if matches else None


def extract_comma_number(text):
    matches = re.findall(r"\$\s*(-?[\d,]+)", text)
    matches = [m for m in matches if m.replace(",", "").strip() != "" and any(c.isdigit() for c in m)]

    if matches:
        last_match = matches[-1].replace(",", "").strip()
        if last_match.lstrip("-").isdigit():
            return int(last_match)
    return None


def read_json(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data

def extract_answer(text):
    """Extracts the answer after '####' in the given text."""
    match = re.search(r'####\s*(-?\d+)', text)
    return match.group(1) if match else None

def extract_word_sorting(text):
    match = re.search(r"####\s*([a-zA-Z\s]+)\n", text)
    if match:
        result = match.group(1).strip()
        return result
    else:
        return None
def get_acc_gsm8k(jsonl_data):

    count=0
    for i in range(len(jsonl_data)):


        answer=int(extract_answer(jsonl_data[i]['answer']))
        jsonl_data[i]['golden_answer'] = answer
        generate_answer = extract_answer(jsonl_data[i]['opt_response'])
        generate_answer_last = extract_last_number(jsonl_data[i]['opt_response'])
        generate_answer_lasttwo = extract_lasttwo_number(jsonl_data[i]['opt_response'])
        generate_answer_lastthree = extract_lasttee_number(jsonl_data[i]['opt_response'])
        generate_answer_number = extract_comma_number(jsonl_data[i]['opt_response'])

        generate_answer_bold = extract_last_bold_number(jsonl_data[i]['opt_response'])

        if generate_answer!=None and int(generate_answer) == int(answer):
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer
            count += 1
        elif generate_answer_last!=None and generate_answer_last == int(answer):
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer_last
            count += 1
        elif generate_answer_lasttwo != None and generate_answer_lasttwo == int(answer):
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer_lasttwo
            count += 1
        elif generate_answer_lastthree != None and generate_answer_lastthree == int(answer):
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer_lastthree
            count += 1
        elif generate_answer_number!=None and generate_answer_number == int(answer):
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer_number
            count += 1
        elif generate_answer_bold!=None and generate_answer_bold == int(answer):
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer_bold
            count += 1

        else:
            jsonl_data[i]['optimized_acc'] =0
            jsonl_data[i]['opt_answer'] = (
                    generate_answer
                    or generate_answer_bold
                    or generate_answer_number
                    or generate_answer_last
                    or generate_answer_lasttwo
                    or generate_answer_lastthree
            )


    return count

def get_acc_word_sorting(save_file,jsonl_data):

    count=0
    for i in range(len(jsonl_data)):
        # print(i)

        answer=jsonl_data[i]['answer']

        generate_answer = extract_word_sorting(jsonl_data[i]['opt_response'])

        if generate_answer!=None and answer in generate_answer:
            jsonl_data[i]['optimized_acc'] = 1
            jsonl_data[i]['opt_answer'] = generate_answer
            count += 1


        else:
            jsonl_data[i]['optimized_acc'] =0
            jsonl_data[i]['opt_answer'] = generate_answer

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(jsonl_data, f, indent=4, ensure_ascii=False)
    return count
def extract_score(text):
    match = re.search(r"Score:\s*(\d+)", text)
    if match:
        score = int(match.group(1))
        return score
    else:
        return None
def get_acc_word_sorting_raw(save_file,jsonl_data):

    count=0
    for i in range(len(jsonl_data)):
        # print(i)

        answer=jsonl_data[i]['answer']
        generate_answer=jsonl_data[i]['raw_response']
        # if jsonl_data[i]['raw_response']:
        #     generate_answer = extract_word_sorting(jsonl_data[i]['raw_response'].strip())

        if generate_answer:
            if answer.strip().lower() in generate_answer.strip().lower():
                jsonl_data[i]['raw_acc'] = 1
                jsonl_data[i]['raw_answer'] = generate_answer
                count += 1


        else:
            jsonl_data[i]['raw_acc'] = 0
            jsonl_data[i]['raw_answer'] = generate_answer

    # with open(save_file, "w", encoding="utf-8") as f:
    #     json.dump(jsonl_data, f, indent=4, ensure_ascii=False)
    return count
def raw(file_path):
    jsonl_data = read_json(file_path)
    count = 0
    for item in jsonl_data:
        if item['raw_acc'] == 1:
            count += 1
    print(f'\nraw acc: {count}/{len(jsonl_data)}={count / len(jsonl_data)}')

def extract_answer_arc(text):
    """Extracts the answer after '####' in the given text."""
    match = re.search(r'##([A-Za-z1-9])', text)
    return match.group(1).upper() if match else None
for llm in ['qwen25_7b']:#qwen3-8b,'qwen3-4b','ds_llama_8b','llama3-1b'
    for task in ['selfeval', 'BPO_test', 'vicuna']:  # 'gsm8k','arc_challenge'

        mepo_file = f"your_result_path{llm}/{task}_poopt_ans_score.json"
        icl_file = f"your_result_path{llm}/{task}_icl_ans_score.json"
        raw_file = f"your_result_path{llm}/{task}_raw_ans_score.json"
        poicl_file = f"your_result_path{llm}/{task}_poicl_ans_score.json"
        rawdata = read_json(raw_file)
        poicldata = read_json(poicl_file)
        icldata = read_json(icl_file)

        # tmpdata = read_json(tmp_file)
        mepodata = read_json(mepo_file)

        optr, rawr, tier = 0, 0, 0
        optb, rawb, tieb = 0, 0, 0
        optt, rawt, tiet = 0, 0, 0
        optp, rawp, tiep = 0, 0, 0
        assert len(rawdata) == len(icldata) == len(mepodata) == len(poicldata)

        for i in range(len(rawdata)):
            # print(i)
            sraw = extract_score(rawdata[i]['score'])

            smepo = extract_score(mepodata[i]['score'])

            sicl = extract_score(icldata[i]['score'])


            spoicl = extract_score(poicldata[i]['score'])

            if sraw > smepo:
                rawr += 1
            elif sraw < smepo:
                optr += 1
            elif sraw == smepo:
                tier += 1

            if sraw > sicl:
                rawp += 1
            elif sraw < sicl:
                optp += 1
            elif sraw == sicl:
                tiep += 1

            if sraw > spoicl:
                rawb += 1
            elif sraw < spoicl:
                optb += 1
            elif sraw == spoicl:
                tieb += 1

        print(
            f'{llm} {task} mepo: mepo: {optr * 100 / len(rawdata)}  tie: {tier * 100 / len(rawdata)}  raw: {rawr * 100 / len(rawdata)}\n')
        print(
            f'{llm} {task} icl: icl: {optp * 100 / len(rawdata)}  tie: {tiep * 100 / len(rawdata)}  raw: {rawp * 100 / len(rawdata)}\n')
        print(
            f'{llm} {task} poicl: poicl: {optb * 100 / len(rawdata)}  tie: {tieb * 100 / len(rawdata)}  raw: {rawb * 100 / len(rawdata)}\n')

        print(len(rawdata))