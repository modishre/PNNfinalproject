# # '''Try running your code with gpt-2 or some smaller model to make sure the syntax works
# # Error is popping up from no device type mentioned so make sure you pass a device map/type when you call insight'''

import os
from nnsight import LanguageModel
import torch
from torch.nn.functional import softmax
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EarlyExitMechanismNDIF:
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold

        # Load API keys from .env
        self.ndif_api_key = os.getenv("NDIF_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

        if not self.ndif_api_key or not self.huggingface_token:
            raise ValueError("NDIF_API_KEY or HUGGINGFACE_TOKEN is missing. Check your .env file.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize the Mistral model with NDIF
        print("Initializing Mistral-7B-Instruct model with NDIF...")
        self.model = LanguageModel(
            self.model_name,
            access_token=self.huggingface_token,
            remote=True,
            ndif_api_key=self.ndif_api_key,
            device_map="auto",
            device=self.device
        )
        print("Model initialized successfully!")

    def run_with_early_exit(self, input_text):
        """
        Run the model with early exit based on confidence threshold.
        """
        try:
            print(f"Input text: {input_text}")
            with self.model.trace(input_text, scan=True, validate=True) as tracer:
                for layer_idx, layer in enumerate(self.model.model.layers):
                    print(f"\nProcessing layer {layer_idx}...")
                    try:
                        # Capture intermediate states
                        hidden_states = layer.output[0].save()
                        logits = self.model.lm_head(hidden_states).save()
                        probabilities = softmax(logits, dim=-1).save()

                        # Debugging outputs
                        print(f"Hidden states shape: {hidden_states.shape if hidden_states is not None else 'None'}")
                        print(f"Logits shape: {logits.shape if logits is not None else 'None'}")
                        print(f"Probabilities shape: {probabilities.shape if probabilities is not None else 'None'}")

                        if hidden_states is None or logits is None or probabilities is None:
                            print(f"Skipping layer {layer_idx} due to missing output.")
                            continue

                        resolved_probs = probabilities.fetch().value
                        confidence, predicted_idx = resolved_probs.max(dim=-1)

                        print(
                            f"Layer {layer_idx}: Confidence = {confidence.item()}, Predicted Index = {predicted_idx.item()}")

                        # Validate confidence and early exit condition
                        if not (0 <= confidence.item() <= 1):
                            print(f"Invalid confidence value: {confidence.item()} at layer {layer_idx}")
                            continue

                        if confidence.item() >= self.confidence_threshold:
                            print(f"Early exit at layer {layer_idx} with confidence {confidence.item()}")
                            return self.model.tokenizer.decode(logits.argmax(dim=-1))

                    except Exception as layer_error:
                        print(f"Error processing layer {layer_idx}: {repr(layer_error)}")
                        continue

                # Fallback if no early exit
                print("\nNo early exit. Performing full inference.")
                final_output = self.model(input_text)
                return self.model.tokenizer.decode(final_output.logits.argmax(dim=-1))

        except Exception as e:
            print(f"Unexpected error: {repr(e)}")
            raise

    def run_baseline(self, input_text):
        """
        Perform baseline inference using all layers.
        """
        try:
            with self.model.trace(input_text) as tracer:
                logits = self.model.output.save()
                probabilities = softmax(logits, dim=-1)
                return {
                    "prediction": logits.argmax().item(),
                    "confidence": probabilities.max().item()
                }
        except Exception as e:
            print(f"Error during baseline inference: {repr(e)}")
            raise e
