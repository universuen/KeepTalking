import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'google/gemma-2b-it'
LEN_ATTACK_PROMPTS = 25

EPOCHS = 100
LR = 0.01

text_prompts = [
    'how are you',
    'What is your name',
    'Tell a story',
]

tokenizer: transformers.GemmaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
model: transformers.GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
model_embedding_layer: torch.nn.Embedding = model.get_input_embeddings()

learnable_tokens = torch.nn.Parameter(torch.randn(LEN_ATTACK_PROMPTS, model_embedding_layer.embedding_dim).cuda())
optimizer = torch.optim.Adam([learnable_tokens], lr=LR)

eos_token_id = tokenizer.eos_token_id

def generate_embedding_seq(
    model: transformers.GemmaForCausalLM,
    eos_token_id: int, 
    input_embeddings: torch.Tensor, 
    max_len: int = 100,
    tokenizer: transformers.GemmaTokenizerFast = None,
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
        if i % 10 == 0:
            print(f'Already generated {i} words!')
    all_logits = torch.cat(all_logits)

    if tokenizer is not None:
        generated_token_ids = all_logits.argmax(dim=-1)
        generated_sentence = tokenizer.decode(generated_token_ids.tolist(), skip_special_tokens=True)
        print(f'Generated sentence: {generated_sentence}')

    return all_logits


for e in range(1, EPOCHS + 1):
    print(f"Epoch {e} started")
    epoch_loss = 0
    for text_prompt in text_prompts:
        optimizer.zero_grad()
        text_ids = tokenizer(text_prompt, return_tensors="pt")['input_ids']
        text_embeddings = model.get_input_embeddings()(text_ids)
        input_embeddings= torch.cat([text_embeddings, learnable_tokens.unsqueeze(0)], dim=1)
        logits = generate_embedding_seq(model, eos_token_id, input_embeddings, tokenizer=tokenizer)

        loss = torch.mean(torch.softmax(logits, dim=1)[:, eos_token_id])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {e} completed with loss: {epoch_loss/len(text_prompts)}")
        
