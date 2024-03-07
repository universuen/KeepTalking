import context

import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.learnable_prompts import LearnablePrompts
import configs


MODEL_NAME = 'google/gemma-2b-it'


if __name__ == '__main__':
    tokenizer: transformers.GemmaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: transformers.GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
    learnable_prompts = LearnablePrompts(5, model_embedding_layer.embedding_dim).to(model.device)

    ids = learnable_prompts.to_ids(model_embedding_layer)
    print(ids)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(' '.join(tokens))
    learnable_prompts.embeddings = nn.Parameter(learnable_prompts.embeddings + 0.1)
    print(learnable_prompts.embeddings)
    learnable_prompts.save(configs.PathConfig().models / 'test.pt')
    del learnable_prompts
    new_lp = LearnablePrompts(5, model_embedding_layer.embedding_dim).to(model.device)
    print(new_lp.embeddings)
    new_lp.load(configs.PathConfig().models / 'test.pt')
    print(new_lp.embeddings)
