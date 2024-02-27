import context

from src.learnable_prompts import LearnablePrompts
from src.logger import Logger
from src.env import Env
from src import utils
import configs

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


training_config = configs.Gemma2bPartialTrainingConfig()
logger_config = configs.LoggerConfig(level='DEBUG')
path_config = configs.PathConfig()
other_config = configs.OtherConfig()

logger = Logger('gemma_2b_partial', logger_config.level, logger_config.path)
logger.info(training_config)
logger.info(logger_config)
logger.info(path_config)
logger.info(other_config)

env = Env()
env.set('lp_token', other_config.gemma_2b_lp_token)

tokenizer: transformers.GemmaTokenizerFast = AutoTokenizer.from_pretrained('google/gemma-2b-it')
eos_token_id = tokenizer.eos_token_id
model: transformers.GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
        'google/gemma-2b-it', 
        device_map="auto"
    )
model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
learnable_prompts = LearnablePrompts(
    num_prompts=training_config.num_prompts,
    num_dims=model_embedding_layer.embedding_dim,
).to(other_config.device)

prompts = utils.read_prompts(path_config.data / 'training_prompts.txt')
val_prompts = utils.read_prompts(path_config.data / 'validation_prompts.txt')
logger.info(f'Loaded {len(prompts)} training prompts and {len(val_prompts)} validation prompts')


optimizer = torch.optim.Adam([learnable_prompts.embeddings], lr=training_config.lr)
logger.info(f'Using Adam')

for e in range(1, training_config.epochs + 1):
    logger.info(f"Epoch {e} started")
    epoch_loss = 0
    for i, text_prompt in enumerate(prompts):
        logger.info(f'[Epoch {e} | {(i + 1) / len(prompts):.2%}] Tuning on prompt: {text_prompt}')
        optimizer.zero_grad()
        input_embeddings = utils.construct_input_embeddings(
            text_prompt=text_prompt,
            tokenizer=tokenizer,
            embedding_layer=model_embedding_layer,
            lp_embeddings=learnable_prompts.embeddings,
        )
        logits = utils.get_last_logits(
                model=model, eos_token_id=eos_token_id, 
                input_embeddings=input_embeddings,
                max_len=training_config.max_len_for_full, 
                tokenizer=tokenizer, logger=logger,
            )
        loss = torch.softmax(logits, dim=1)[:, eos_token_id]
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        logger.debug(f'Loss: {loss.item()}')
        if (i + 1) % 1 == 0:
            logger.info('Evaluating')
            torch.cuda.empty_cache()
            avg_len = utils.evaluate(
                model = model,
                learnable_prompts=learnable_prompts,
                val_prompts=val_prompts,
                tokenizer=tokenizer,
                logger=logger,
            )
            logger.info(f'Avg response length on val dataset: {avg_len}')

    logger.info(f"Epoch {e} completed with loss: {epoch_loss/len(prompts)}")

ids = learnable_prompts.to_ids(model_embedding_layer)
lp_text = tokenizer.decode(ids)
logger.info(f"Learned prompts: {lp_text}")
