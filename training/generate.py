import torch
from config import config


def generate_text(model, tokenizer, seed_text, max_length=50):

    model.eval()

    tokens = tokenizer.encode(seed_text)

    device = torch.device(config.device)

    tokens = tokens.copy()

    for _ in range(max_length):

        input_seq = tokens[-config.seq_length :]

        if len(input_seq) < config.seq_length:
            break

        x = torch.tensor([input_seq], dtype=torch.long).to(device)

        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, 1).item()

        tokens.append(next_token)

    return tokenizer.decode(tokens)
