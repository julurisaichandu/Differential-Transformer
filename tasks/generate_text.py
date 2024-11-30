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


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Make sure that checkpoint model params are same.
model_params = {
    "vocab_size": tokenizer.vocab_size,
    "d_model": 512,
    "n_heads": 8,
    "d_head": 64,
    "n_layers": 6,
    "max_seq_len": 128,
    "dropout": 0.1
}


checkpoint_path = "checkpoint_trial_2_epoch_5.pt"  # Adjust this to your checkpoint path
model = load_model(checkpoint_path, model_params)


prompt = "We live in the word of desires but"
generated_text = generate_text(model, tokenizer, prompt, max_length=5)
print(f"Generated text:\n{generated_text}")