import torch
from tokenizers import Tokenizer
from Multi_Head_Diff_Transformer import EncoderDecoderTransformer
from tqdm import tqdm


class SentencePredictor:
    def __init__(self, model_path, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = 128

        self.model = EncoderDecoderTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_head=64,
            d_ff=2048,
            max_seq_len=self.max_length,
            dropout=0.1
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['train_loss']:.4f}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    def _pad_sequence(self, seq):
        if len(seq) < self.max_length:
            return seq + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - len(seq))
        return seq[:self.max_length]

    def generate_completion(self, input_text, max_length=50, temperature=0.7):
        self.model.eval()

        encoded = self.tokenizer.encode(input_text)
        input_tokens = self._pad_sequence(encoded.ids)
        input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(self.device)

        bos_token_id = self.tokenizer.token_to_id("[BOS]")
        target_tokens = torch.tensor([[bos_token_id]]).to(self.device)

        generated_tokens = []

        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="Generating"):
                padded_target = self._pad_sequence(target_tokens[0].tolist())
                padded_target = torch.tensor(padded_target).unsqueeze(0).to(self.device)

                output = self.model(input_tokens, padded_target)

                next_token_logits = output[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)

                next_token = torch.multinomial(next_token_probs, num_samples=1)

                if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                    break

                generated_tokens.append(next_token.item())
                target_tokens = torch.cat([target_tokens, next_token], dim=1)

        completed_text = self.tokenizer.decode(generated_tokens)
        return completed_text


def test_model():
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        return

    try:
        predictor = SentencePredictor(
            model_path="checkpoint_epoch_2.pt",
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"Error creating predictor: {str(e)}")
        return

    test_sentences = [
        "The quick brown fox",
        "In the middle of",
        "She opened the door and",
        "The scientists discovered that",
        "Despite the challenging circumstances"
    ]

    print("\nTesting sentence completions:")
    print("-" * 50)

    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        try:
            completion = predictor.generate_completion(
                sentence,
                max_length=50,
                temperature=0.7
            )
            print(f"Completion: {sentence} {completion}")
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    test_model()