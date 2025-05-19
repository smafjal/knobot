from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


@dataclass
class InferenceConfig:
    model_path: str = "./models"
    max_length: int = 512
    num_beams: int = 4
    early_stopping: bool = True


class ModelInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model: Optional[T5ForConditionalGeneration] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
    def load_model(self) -> None:
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path {model_path} does not exist")
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _prepare_input(self, instruction: str) -> torch.Tensor:
        input_text = f"instruction: {instruction}"
        return self.tokenizer(
            input_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def generate_response(self, instruction: str) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before inference")
            
        try:
            # Prepare input
            inputs = self._prepare_input(instruction)
            
            # Generate response
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping
            )
            
            # Decode and return response
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            raise RuntimeError(f"Error during inference: {str(e)}")


class InteractiveInference:
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.inference = ModelInference(self.config)
        
    def start_session(self) -> None:
        try:
            print("Loading model...")
            self.inference.load_model()
            print("Model loaded successfully!")
            
            while True:
                instruction = input("\nEnter your instruction (or 'quit' to exit): ").strip()
                
                if not instruction:
                    continue
                    
                if instruction.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                try:
                    response = self.inference.generate_response(instruction)
                    print("\nResponse:", response)
                except Exception as e:
                    print(f"\nError: {str(e)}")
                    
        except Exception as e:
            print(f"Fatal error: {str(e)}")


def main():
    inference = InteractiveInference()
    inference.start_session()


if __name__ == "__main__":
    main()
