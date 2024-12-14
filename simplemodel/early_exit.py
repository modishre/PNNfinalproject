from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class EarlyExitRunner:
    def __init__(self, model_name):
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Default confidence threshold
        self.threshold = 0.8

    def set_threshold(self, threshold):
        """
        Set the confidence threshold for early exit.
        """
        self.threshold = threshold

    def run_with_early_exit(self, dataset):
        """
        Run inference with an early exit mechanism based on confidence thresholds.
        """
        results = []

        for idx, sample in enumerate(dataset):
            input_text = sample['text']

            # Tokenize input text with padding and truncation
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Forward pass to get hidden states and logits
            output = self.model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states
            logits = output.logits

            # Early exit logic: Check confidence at each layer
            for layer_idx, hidden_state in enumerate(hidden_states):
                layer_logits = self.model.lm_head(hidden_state)
                probs = torch.nn.functional.softmax(layer_logits, dim=-1)
                token_confidences, predicted_indices = torch.max(probs, dim=-1)

                # Use the average confidence across all tokens as a scalar measure
                avg_confidence = token_confidences.mean().item()

                # Check if confidence exceeds the threshold
                if avg_confidence >= self.threshold:
                    print(f"Exiting early at layer {layer_idx + 1} with confidence {avg_confidence}")
                    predicted_text = self.tokenizer.decode(predicted_indices[0], skip_special_tokens=True)
                    results.append({
                        "text": input_text,
                        "predicted_text": predicted_text,
                        "confidence": avg_confidence,
                        "layer": layer_idx + 1
                    })
                    break
            else:
                # If no early exit, perform full inference
                print(f"No early exit for: {input_text}. Performing full inference.")
                predicted_indices = torch.argmax(logits, dim=-1)
                predicted_text = self.tokenizer.decode(predicted_indices[0], skip_special_tokens=True)
                avg_confidence = token_confidences.mean().item()  # Confidence from final layer
                results.append({
                    "text": input_text,
                    "predicted_text": predicted_text,
                    "confidence": avg_confidence,
                    "layer": len(hidden_states)
                })

            # Print progress for debugging purposes
            if idx % 10 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples.")

            # Stop after a fixed number of samples for debugging purposes (optional)
            if idx >= 100:  # Process only 100 samples for now
                print("Stopping early for debugging purposes.")
                break

        # Return collected results
        return results
