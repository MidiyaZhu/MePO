# 🚀 MePO: Merit-Guided Prompt Optimization （EACL 2026 Main Paper)

We introduce **MePO**, a lightweight and locally deployable prompt optimization model trained under a **merits-guided preference framework**. MePO is designed to optimize prompts effectively for downstream use in small language models.

Real-time application video:
https://youtu.be/mDQtaJEKB2o

## 📚 Dataset

The dataset used for training and evaluation is available on Hugging Face:

- [**MePO**](https://huggingface.co/datasets/zixiaozhu/MePO)  
- [**MePO_BPO**](https://huggingface.co/datasets/zixiaozhu/MePO_BPO) — Optimized prompts based on the BPO dataset  
- [**MePO_Alpaca**](https://huggingface.co/datasets/zixiaozhu/MePO_Alpaca) — Optimized prompts based on the Alpaca dataset

## 📚 Model:  
[**zixiaozhu/MePO**](https://huggingface.co/zixiaozhu/MePO)

## 🛠️ Implementation

To train your own prompt optimization model using MePO, simply run with downloaded dataset in your correct folder path:
```bash
pip install -r requirements.txt


python MePO_run_train.py
```
> 📌 **Recommendation:**  
Based on our empirical results, we recommend using [**MePO_BPO**](https://huggingface.co/datasets/zixiaozhu/MePO_BPO) for training prompt optimizers targeting **lightweight LLMs (<7B)**, especially in chatbot-style prompt optimization tasks.

### **For chatbot-style testing demonstration:**
```bash
MePO_prompt_optimization.py
```

### **For downstream tasks optimization prompt generation:**
```bash
MePO_optimized_downstream_task.py  (gsm8k,bbh,arc-c,arc-e,piqa)

('selfeval', 'BPO_test', 'vicuna')
cd ./instruction-following

Inference_downstream_instrucionfollowing.py (po is mepo prompt, raw is original prompt from HuggingFace dataset)

evaluation_instruction_score_deepseek.py (Scoring)

evaluation_instruction_winrate.py (Winrate Comparison)
```

## 📄 Citation

If you use our code, dataset, or model, please cite our paper (accept as **EACL 2026 Main**):

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
