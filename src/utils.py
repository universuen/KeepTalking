from pathlib import Path

import torch
import transformers

from src.logger import Logger
from src.env import Env


env = Env()


def read_prompts(file_path: Path):
    with open(file_path, 'r') as f:
        return [i.strip('\n') for i in f.readlines()]


def generate_logits_seq(
    model,
    eos_token_id: int, 
    input_embeddings: torch.Tensor, 
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
):
    model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
    all_logits = []
    for _ in range(1, max_len + 1):
        outputs = model(inputs_embeds=input_embeddings)
        logits = outputs.logits
        all_logits.append(logits[:, -1, :]) 
        next_token_id = logits[:, -1, :].argmax(dim=1).unsqueeze(1)
        if next_token_id == eos_token_id:
            break
        next_token_embedding = model_embedding_layer(next_token_id)
        input_embeddings = torch.cat([input_embeddings, next_token_embedding], dim=1)
    all_logits = torch.cat(all_logits)

    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return all_logits


@torch.no_grad()
def evaluate_by_embeddings(model, learnable_prompts, val_prompts, tokenizer, max_len=100, logger=None):
    generated_lengths = []
    for i in val_prompts:
        if logger is not None:
            logger.debug(f'Evaluating on prompt:{i}')
        input_embeddings = construct_input_embeddings(
            text_prompt=i, 
            tokenizer=tokenizer, 
            embedding_layer=model.get_input_embeddings(),
            lp_embeddings=learnable_prompts.embeddings,
        )
        logits = generate_logits_seq(
            model,
            eos_token_id=tokenizer.eos_token_id, 
            input_embeddings=input_embeddings, 
            max_len=max_len,
        )
        generated_token_ids = logits.argmax(dim=-1)
        generated_text = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result:{generated_text}')
        generated_lengths.append(len(generated_text.split()))
        logger.debug(f'Length: {len(generated_text.split())}')
    average_length = sum(generated_lengths) / len(generated_lengths)
    return average_length


def get_last_logits(
    model,
    eos_token_id: int, 
    input_embeddings: torch.Tensor, 
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
) -> torch.Tensor:
    model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
    all_logits = []
    last_logits = None
    for i in range(1, max_len + 1):
        with torch.no_grad():
            outputs = model(inputs_embeds=input_embeddings)
        logits = outputs.logits
        all_logits.append(logits[:, -1, :]) 
        next_token_id = logits[:, -1, :].argmax(dim=1).unsqueeze(1)
        if next_token_id == eos_token_id or i == max_len:
            outputs = model(inputs_embeds=input_embeddings)
            last_logits = outputs.logits[:, -1, :]
            break
        next_token_embedding = model_embedding_layer(next_token_id)
        input_embeddings = torch.cat([input_embeddings, next_token_embedding], dim=1)
    all_logits = torch.cat(all_logits)

    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return last_logits


@torch.no_grad()
def evaluate(model, learnable_prompts, val_prompts, tokenizer, max_len=100, logger=None):
    lp_ids = learnable_prompts.to_ids(model.get_input_embeddings())
    generated_lengths = []
    for i in val_prompts:
        if logger is not None:
            logger.debug(f'Evaluating on prompt:{i}')
        input_ids = construct_input_ids(
            text_prompt=i,
            tokenizer=tokenizer,
            lp_ids=lp_ids
        )
        output = model.generate(input_ids, max_new_tokens=max_len)
        generated_text = tokenizer.decode(output[0]).replace(tokenizer.decode(input_ids.flatten()), '', 1)
        if logger is not None:
            logger.debug(f'Corresponding result:{generated_text}')
        generated_lengths.append(len(generated_text.split()))
        logger.debug(f'Length: {len(generated_text.split())}')
    average_length = sum(generated_lengths) / len(generated_lengths)
    return average_length


def decorate_tokenizer(tokenizer: transformers.PreTrainedTokenizerFast) -> None:
    if env.get('lp_token') not in tokenizer.added_tokens_encoder.keys():
        tokenizer.add_tokens(env.get('lp_token'), special_tokens=True)


def construct_input_embeddings(
        text_prompt: str, 
        tokenizer: transformers.PreTrainedTokenizerFast, 
        embedding_layer: torch.nn.Embedding,
        lp_embeddings: torch.Tensor,
    ) -> torch.Tensor:
    decorate_tokenizer(tokenizer)
    lp_id = tokenizer.added_tokens_encoder[env.get('lp_token')]
    prompt = f"{text_prompt} {env.get('lp_token')}"
    encoded_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    cut_off_point = encoded_ids.index(lp_id)
    prefix_ids = torch.tensor([encoded_ids[:cut_off_point]], dtype=torch.long, device=lp_embeddings.device)
    suffix_ids = torch.tensor([encoded_ids[cut_off_point + 1:]], dtype=torch.long, device=lp_embeddings.device)
    prefix_embeddings = embedding_layer(prefix_ids)
    suffix_embeddings = embedding_layer(suffix_ids)
    constructed_embeddings = torch.cat(
        [prefix_embeddings, lp_embeddings.unsqueeze(0), suffix_embeddings],
        dim=1,
    )
    return constructed_embeddings


@torch.no_grad()
def construct_input_ids(
        text_prompt: str, 
        tokenizer: transformers.PreTrainedTokenizerFast, 
        lp_ids: torch.Tensor,
    ) -> torch.Tensor:
    decorate_tokenizer(tokenizer)
    lp_id = tokenizer.added_tokens_encoder[env.get('lp_token')]
    prompt = f"{text_prompt} {env.get('lp_token')}"
    encoded_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    cut_off_point = encoded_ids.index(lp_id)
    prefix_ids = torch.tensor([encoded_ids[:cut_off_point]], dtype=torch.long, device=lp_ids.device)
    suffix_ids = torch.tensor([encoded_ids[cut_off_point + 1:]], dtype=torch.long, device=lp_ids.device)
    constructed_ids = torch.cat([prefix_ids, lp_ids.unsqueeze(0), suffix_ids], dim=1)
    return constructed_ids
