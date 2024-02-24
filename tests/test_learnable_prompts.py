import context

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.learnable_prompts import LearnablePrompts


MODEL_NAME = 'google/gemma-2b-it'


if __name__ == '__main__':
    tokenizer: transformers.GemmaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: transformers.GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
    learnable_prompts = LearnablePrompts(5, model_embedding_layer.embedding_dim).to(model.device)

    ids = learnable_prompts.to_ids(model_embedding_layer.weight.data)
    print(ids)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(' '.join(tokens))
