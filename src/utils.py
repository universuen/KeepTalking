from pathlib import Path
from typing import Callable
from functools import partial

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
    input_embeddings: torch.Tensor, 
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
):
    generate_with_grad = enable_grad(model.generate)
    outputs = generate_with_grad(
        model, 
        inputs_embeds=input_embeddings, 
        max_new_tokens=max_len, 
        return_dict_in_generate=True, 
        output_logits=True,
    )
    all_logits = torch.cat(outputs.logits)
    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return all_logits


def get_last_logits(
    model,
    eos_token_id: int, 
    input_embeddings: torch.Tensor, 
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
) -> torch.Tensor:
    model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
    outputs = model.generate(
        inputs_embeds=input_embeddings, 
        max_new_tokens=max_len, 
        return_dict_in_generate=True,
        output_logits=True,
    )
    all_logits = torch.cat(outputs.logits)
    input_ids = outputs.sequences
    try:
        eos_index = input_ids[0].tolist().index(eos_token_id)
    except ValueError:
        eos_index = -1
    target_input_ids = input_ids[:, :eos_index]
    response_embs = model_embedding_layer(target_input_ids)
    inputs_embs = torch.cat([input_embeddings, response_embs], dim=1)
    last_logits = model(inputs_embeds=inputs_embs).logits[:, -1, :]
    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return last_logits


def generate_logits_seq_blip2(
    model,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
):
    generate_with_grad = enable_grad(model.generate)
    lang_model_generate_with_grad = enable_grad(model.language_model.generate)
    original_language_model_generate = model.language_model.generate
    model.language_model.generate = partial(lang_model_generate_with_grad, model.language_model)
    outputs = generate_with_grad(
        model, 
        pixel_values=image,
        input_ids=input_ids, 
        max_new_tokens=max_len, 
        return_dict_in_generate=True, 
        output_logits=True,
    )
    model.language_model.generate = original_language_model_generate
    all_logits = torch.cat(outputs.logits)
    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return all_logits


def generate_logits_seq_blip(
    model,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
):
    generate_with_grad = enable_grad(model.generate)
    lang_model_generate_with_grad = enable_grad(model.text_decoder.generate)
    original_text_decoder_generate = model.text_decoder.generate
    model.text_decoder.generate = partial(lang_model_generate_with_grad, model.text_decoder)
    outputs = generate_with_grad(
        model, 
        pixel_values=image,
        input_ids=input_ids, 
        max_new_tokens=max_len, 
        return_dict_in_generate=True, 
        output_logits=True,
    )
    model.text_decoder.generate = original_text_decoder_generate
    all_logits = torch.cat(outputs.logits)
    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return all_logits


def generate_last_logits_blip2(
    model,
    eos_token_id: int, 
    image: torch.Tensor,
    input_ids: torch.Tensor,
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
):
    outputs = model.generate(
        pixel_values=image,
        input_ids=input_ids, 
        max_new_tokens=max_len, 
        return_dict_in_generate=True,
        output_logits=True,
    )
    all_logits = torch.cat(outputs.logits)
    input_ids = outputs.sequences
    try:
        eos_index = input_ids[0].tolist().index(eos_token_id)
    except ValueError:
        eos_index = -1
    target_input_ids = input_ids[:, :eos_index]
    last_logits = generate_logits_seq_blip2(
        model,
        image=image,
        input_ids=target_input_ids,
        max_len=max_len,
    )[-1].unsqueeze(0)
    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result: {generated_sentence}')

    return last_logits



def generate_logits_seq_llava(
    model,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    max_len: int = 100,
    tokenizer = None,
    logger: Logger = None,
):
    generate_with_grad = enable_grad(model.generate)
    outputs = generate_with_grad(
        model, 
        pixel_values=image,
        input_ids=input_ids, 
        max_new_tokens=max_len, 
        return_dict_in_generate=True, 
        output_logits=True,
    )
    all_logits = torch.cat(outputs.logits)
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


@torch.no_grad()
def evaluate_blip2(model, learnable_visual_prompts, val_prompts, tokenizer, max_len=100, logger=None):
    generated_lengths = []
    for i in val_prompts:
        if logger is not None:
            logger.debug(f'Evaluating on prompt:{i}')
        input_ids = tokenizer.encode(f'Question:{i} Answer:', return_tensors='pt').to(model.device)
        output = model.generate(pixel_values=learnable_visual_prompts.embeddings, input_ids=input_ids, max_new_tokens=max_len)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        if logger is not None:
            logger.debug(f'Corresponding result:{generated_text}')
        generated_lengths.append(len(generated_text.split()))
        logger.debug(f'Length: {len(generated_text.split())}')
    average_length = sum(generated_lengths) / len(generated_lengths)
    return average_length


@torch.no_grad()
def evaluate_blip(model, learnable_visual_prompts, val_prompts, tokenizer, max_len=100, logger=None):
    return evaluate_blip2(model, learnable_visual_prompts, val_prompts, tokenizer, max_len, logger)


@torch.no_grad()
def evaluate_llava(model, learnable_visual_prompts, val_prompts, tokenizer, max_len=100, logger=None):
    generated_lengths = []
    for i in val_prompts:
        if logger is not None:
            logger.debug(f'Evaluating on prompt:{i}')
        input_ids = construct_input_ids_llava(i, tokenizer).to(model.device)
        output = model.generate(pixel_values=learnable_visual_prompts.embeddings, input_ids=input_ids, max_new_tokens=max_len)
        input_text = tokenizer.decode(input_ids.flatten(), skip_special_tokens=True).replace('<image>', '')
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text, '', 1)
        if logger is not None:
            logger.debug(f'Corresponding result:{generated_text}')
        generated_lengths.append(len(generated_text.split()))
        logger.debug(f'Length: {len(generated_text.split())}')
    average_length = sum(generated_lengths) / len(generated_lengths)
    return average_length


def decorate_tokenizer(tokenizer: transformers.PreTrainedTokenizerFast) -> None:
    if env.get('lp_token') not in tokenizer.added_tokens_encoder.keys():
        tokenizer.add_tokens(env.get('lp_token'), special_tokens=True)


def construct_input_ids_blip2(
    text_prompt: str, 
    tokenizer,
) -> torch.Tensor:
    prompt = f'Question: {text_prompt} Answer:'
    encoded_ids = tokenizer.encode(prompt, return_tensors="pt")
    return encoded_ids


def construct_input_ids_blip(
    text_prompt: str, 
    tokenizer,
) -> torch.Tensor:
    encoded_ids = tokenizer.encode(text_prompt, return_tensors="pt")
    return encoded_ids

def construct_input_ids_llava(
    text_prompt: str, 
    tokenizer,
) -> torch.Tensor:
    text_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
    encoded_ids = tokenizer.encode(text_prompt, return_tensors="pt")
    return encoded_ids


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


def enable_grad(func: Callable) -> Callable:
    return func.__closure__[1].cell_contents