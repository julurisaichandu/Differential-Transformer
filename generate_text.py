import torch
from transformers import AutoTokenizer
from diff_decoder_only import DecoderOnlyModel


def load_model(checkpoint_path, model_params):
    model = DecoderOnlyModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text
