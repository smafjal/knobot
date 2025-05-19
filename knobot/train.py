from dataclasses import dataclass
from typing import Dict, Any

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from datapipiline.slack import SlackDataPipeline


@dataclass
class TrainingConfig:
    model_name: str = "google/flan-t5-small"
    output_dir: str = "./modeles"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 3
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 50
    max_length: int = 512


class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def _setup_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _setup_model(self) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        
    def _tokenize_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        input_text = f"instruction: {example['instruction']}"
        target_text = example['response']
        
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
        )
        
        labels = self.tokenizer(
            target_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_dataset(self, training_data: list) -> Dataset:
        dataset = Dataset.from_list(training_data)
        return dataset.map(self._tokenize_example, batched=False)
    
    def _setup_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            logging_steps=self.config.logging_steps,
            prediction_loss_only=True
        )
    
    def train(self, training_data: list) -> None:
        # Initialize components
        self._setup_tokenizer()
        self._setup_model()
        
        # Prepare dataset
        tokenized_dataset = self.prepare_dataset(training_data)
        
        # Setup trainer
        training_args = self._setup_training_args()
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        
        # Execute training
        self.trainer.train()
        
        # Save the model and tokenizer
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


def main():
    # Initialize data pipeline
    pipeline = SlackDataPipeline(slack_data=None)
    training_data = pipeline.process_data()
    
    # Setup and execute training
    config = TrainingConfig()
    trainer = ModelTrainer(config)
    trainer.train(training_data)


if __name__ == "__main__":
    main()
