import context

from src.learnable_prompts import LearnablePrompts
from src.logger import Logger
from src import utils
import configs

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


training_config = configs.Gemma2bTrainingConfig()
logger_config = configs.LoggerConfig()
path_config = configs.PathConfig()
other_config = configs.OtherConfig()

logger = Logger('gemma_2b_full', logger_config.level, logger_config.path)
logger.info(training_config)
logger.info(logger_config)
logger.info(path_config)
logger.info(other_config)

tokenizer: transformers.GemmaTokenizerFast = AutoTokenizer.from_pretrained('google/gemma-2b-it')
model: transformers.GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
        'google/gemma-2b-it', 
        device_map="auto"
    )
model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
learnable_prompts = LearnablePrompts(
    num_prompts=training_config.num_prompts,
    num_dims=model_embedding_layer.embedding_dim,
).to(other_config.device)
prompts = utils.read_prompts(path_config.data / 'prompts.txt')
logger.info(f'Loaded {len(prompts)} prompts')
eos_token_id = tokenizer.eos_token_id
prefix = '<start_of_turn>user\n'
suffix = '<end_of_turn><start_of_turn>model\n'
suffix_embedding = model_embedding_layer(
        torch.tensor([tokenizer(suffix)['input_ids'][1:]]).to(other_config.device)
    ).detach()

optimizer = torch.optim.Adam([learnable_prompts.embeddings], lr=training_config.lr)
logger.info(f'Using Adam')

for e in range(1, training_config.epochs + 1):
    logger.info(f"Epoch {e} started")
    epoch_loss = 0
    for prompt in prompts:
        logger.info(f'Tuning on prompt: {prompt}')
        optimizer.zero_grad()
        prompt = prefix + prompt
        text_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
        text_embeddings = model_embedding_layer(text_ids).to(other_config.device)
        input_embeddings= torch.cat(
                [
                    text_embeddings, 
                    learnable_prompts.embeddings.unsqueeze(0),
                    suffix_embedding
                ], 
                dim=1
            )
        logits = utils.generate_logits_seq(
                model=model, eos_token_id=eos_token_id, 
                input_embeddings=input_embeddings,
                max_len=training_config.max_len_for_full, 
                tokenizer=tokenizer, logger=logger,
            )
        loss = torch.mean(torch.softmax(logits, dim=1)[:, eos_token_id])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    logger.info(f"Epoch {e} completed with loss: {epoch_loss/len(prompts)}")

ids = learnable_prompts.to_ids(model_embedding_layer)
tokens = tokenizer.convert_ids_to_tokens(ids)
logger.info(f"Learned prompts: {' '.join(tokens)}")
