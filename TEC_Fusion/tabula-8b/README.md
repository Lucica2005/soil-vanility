---
license: llama3
datasets:
- jpgard/t4-full
language:
- en
---

This repository contains the TabuLa-8B (Tabular Llama-8B) model. 
TabuLa-8B is a foundation model for prediction (classification and binned regression) on tabular data.

TabuLa-8B is described in the paper ["Large Scale Transfer Learning for Tabular Data via Language Modeling."](https://arxiv.org/abs/2406.12031)

For more details on the model, see the paper, which includes a Model Card detailing the model architecture, training, and evaluation.
TabuLa-8B was trained with [rtfm](https://github.com/mlfoundations/rtfm), 
using the [T4 dataset](https://huggingface.co/datasets/mlfoundations/t4-full).

TabuLa-8B is built with Meta Llama 3.

# Usage and Examples

You can load the model with `transformers` via
```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mlfoundations/tabula-8b")
model = AutoModelForCausalLM.from_pretrained("mlfoundations/tabula-8b")
```

For more information on how to prepare data and run inference (including a demo notebook for performing inference on your data), see the examples in [rtfm](https://github.com/mlfoundations/rtfm).

# License and Terms of Use

TabuLa-8B is fine-tuned from the Llama-3 8B model. 
As a result, we release it under the [Llama 3 license](https://llama.meta.com/llama3/license/), 
and by using the model you agree to abide by the [Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/) 
and the Llama 3 [Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).
