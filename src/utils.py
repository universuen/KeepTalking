from pathlib import Path

import torch

from src.logger import Logger


def read_prompts(file_path: Path):
    with open(file_path, 'r') as f:
        return [i.strip('\n') for i in f.readlines()]


def generate_logits_seq(
    model,
    eos_token_id: int, 
    input_embeddings: torch.Tensor, 
    max_len: int = 100,
    tokenizer = None,
    logger = None,
):
    model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
    all_logits = []
    for i in range(1, max_len + 1):
        outputs = model(inputs_embeds=input_embeddings)
        logits = outputs.logits
        all_logits.append(logits[:, -1, :]) 
        next_token_id = logits[:, -1, :].argmax(dim=1).unsqueeze(1)
        if next_token_id == eos_token_id:
            break
        next_token_embedding = model_embedding_layer(next_token_id)
        input_embeddings = torch.cat([input_embeddings, next_token_embedding], dim=1)
        if logger is not None:
            if i % 10 == 0:
                logger.info(f'Already generated {i} words!')
    all_logits = torch.cat(all_logits)

    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.info(f'Generated sentence: {generated_sentence}')

    return all_logits