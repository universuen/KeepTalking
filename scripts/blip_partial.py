import context

from src.models.learnable_visual_prompts import LearnableVisualPrompts
from src.logger import Logger
from src import utils
import configs

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering


training_config = configs.BlipPartialTrainingConfig()
logger_config = configs.LoggerConfig(level='DEBUG')
path_config = configs.PathConfig()
other_config = configs.OtherConfig(device='cuda:1')

logger = Logger('blip_partial', logger_config.level, logger_config.path)
logger.info(training_config)
logger.info(logger_config)
logger.info(path_config)
logger.info(other_config)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large", device_map=other_config.device)
tokenizer = processor.tokenizer
eos_token_id = model.config.text_config.sep_token_id

model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()
learnable_prompts = LearnableVisualPrompts(
    num_channels=3,
    height=processor.image_processor.size['height'],
    width=processor.image_processor.size['width'],
).to(model.device)

prompts = utils.read_prompts(path_config.data / 'training_prompts.txt')
val_prompts = utils.read_prompts(path_config.data / 'validation_prompts.txt')
logger.info(f'Loaded {len(prompts)} training prompts and {len(val_prompts)} validation prompts')


optimizer = torch.optim.Adam([learnable_prompts.embeddings], lr=training_config.lr)
logger.info(f'Using Adam')

for e in range(1, training_config.epochs + 1):
    logger.info(f"Epoch {e} started")
    epoch_loss = 0
    step_cnt = 0
    for i in range(0, len(prompts), training_config.batch_size):
        loss = 0
        optimizer.zero_grad()
        for text_prompt in prompts[i:i + training_config.batch_size]:
            logger.info(f'[Epoch {e}] Tuning on prompt: {text_prompt}')
            input_ids = utils.construct_input_ids_blip(text_prompt, tokenizer).to(model.device)
            logits = utils.generate_last_logits_blip(
                model=model,
                eos_token_id=eos_token_id,
                image=learnable_prompts.embeddings,
                input_ids=input_ids,
                max_len=training_config.max_len,
                tokenizer=tokenizer,
                logger=logger,
            )
            sub_loss = torch.softmax(logits, dim=1)[:, eos_token_id] / training_config.batch_size
            sub_loss.backward()
            loss += sub_loss.item()
        optimizer.step()
        learnable_prompts.normalize()
        step_cnt += 1
        epoch_loss += loss
        logger.debug(f'Loss: {loss}')
        if step_cnt % 1 == 0:
            logger.info('Evaluating')
            torch.cuda.empty_cache()
            avg_len = utils.evaluate_blip(
                model = model,
                learnable_visual_prompts=learnable_prompts,
                val_prompts=val_prompts,
                tokenizer=tokenizer,
                logger=logger,
            )
            logger.info(f'Avg response length on val dataset: {avg_len}')

    logger.info(f"Epoch {e} completed with loss: {epoch_loss * training_config.batch_size / len(prompts)}")

img_saving_path = path_config.data / 'blip_partial.png'
pil_img = learnable_prompts.to_image()
pil_img.save(img_saving_path)
logger.info(f'Learned image is saved at {img_saving_path}')

embeddings_saving_path = path_config.models / 'blip_partial.pt'
learnable_prompts.save(embeddings_saving_path)
logger.info(f'Learned embeddings are saved at {embeddings_saving_path}')
