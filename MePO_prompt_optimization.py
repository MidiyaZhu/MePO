from transformers import AutoModelForCausalLM, AutoTokenizer


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()


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



with open("./data/optimize_prompt_instrution.txt", "r",
          encoding="utf-8") as file:
    po_prompt_ins = file.read()


print("Loading PO model...")

po_model_path = "./model/"

model2, tokenizer2 = load_model_and_tokenizer(po_model_path)


# Loop for user input
while True:
    prompt = input("Enter your prompt (type 'exit' to quit): ")
    if prompt.strip().lower() == "exit":
        print("Exiting...")
        break
    print(f"##################\nRaw Prompt: {prompt}")


    # Optimize prompt with PO model
    po_qs_input = po_prompt_ins.replace("S_P", prompt)
    po_opt_prompt = generate_response(model2, tokenizer2, po_qs_input)
    print(f"\n--- Optimized Prompt by PO Model ---\n{po_opt_prompt}")


    print("\n========================================\n")

