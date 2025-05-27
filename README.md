# 🚀 MePO: Merit-Guided Prompt Optimization

We introduce **MePO**, a lightweight and locally deployable prompt optimization model trained under a **merits-guided preference framework**. MePO is designed to optimize prompts effectively for downstream use in small language models.

## 📚 Dataset

The dataset used for training and evaluation can be found at [**zixiaozhu/MePO**](https://huggingface.co/datasets/zixiaozhu/MePO/tree/main), with two subsets:

- [**MePO_BPO**](https://huggingface.co/datasets/zixiaozhu/MePO_BPO) — Optimized prompts based on the BPO dataset  
- [**MePO_Alpaca**](https://huggingface.co/datasets/zixiaozhu/MePO_Alpaca) — Optimized prompts based on the Alpaca dataset

The MePO model will be released on Hugging Face soon.

## 🛠️ Implementation

To train your own prompt optimization model using MePO, simply run with downloaded dataset in your correct folder path:
```bash
pip install -r requirements.txt


python MePO_run_train.py
```
> 📌 **Recommendation:**  
Based on our empirical results, we recommend using [**MePO_BPO**](https://huggingface.co/datasets/zixiaozhu/MePO_BPO) for training prompt optimizers targeting **lightweight LLMs (<7B)**, especially in chatbot-style prompt optimization tasks.

For chatbot-style testing demonstration:
```bash
MePO_prompt_optimization.py
```

For downstream tasks optimization prompt generation:
```bash
MePO_optimized_downstream_task.py
```

## 📄 Citation

If you use our code, dataset, or model, please cite our paper:

```bibtex
@misc{zhu2025rethinkingpromptoptimizersprompt,
  title     = {Rethinking Prompt Optimizers: From Prompt Merits to Optimization},
  author    = {Zixiao Zhu and Hanzhang Zhou and Zijian Feng and Tianjiao Li and Chua Jia Jim Deryl and Mak Lee Onn and Gee Wah Ng and Kezhi Mao},
  year      = {2025},
  eprint    = {2505.09930},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2505.09930}
}
