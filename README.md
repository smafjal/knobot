# Knobot - Question Answering System

A production-grade question answering system powered by RAG (Retrieval-Augmented Generation) and local language models.

## Features

- FastAPI-based REST API
- RAG-based document retrieval
- Local language model inference
- Production-grade error handling and logging
- Configuration management
- Health check endpoints
- CORS support
- Custom model training capabilities
- Interactive inference mode

## Prerequisites

- Python 3.8+
- pip
- virtualenv (recommended)
- CUDA-capable GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/knobot.git
cd knobot
```

2. Create and activate a virtual environment:
```bash
./setup_venv.sh
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The application can be configured using environment variables or a `.env` file. Create a `.env` file in the project root with the following variables:

```env
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Usage

### Training the Model

1. Prepare your training data in the required format
2. Run the training script:
```bash
python -m knobot.train
```

The trained model will be saved in the `models/` directory.

### Interactive Inference

Run the interactive inference mode:
```bash
python -m knobot.inference
```

This will start an interactive session where you can input questions and get responses from the model.

### API Server

1. Start the API server:
```bash
python -m knobot.api
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /ask`: Ask a question
  ```json
  {
    "text": "Your question here"
  }
  ```

- `POST /documents`: Add documents to the RAG system
  ```json
  {
    "documents": ["Document 1", "Document 2"]
  }
  ```

- `GET /health`: Health check endpoint

## Development

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

Run the development tools:
```bash
black knobot/
isort knobot/
mypy knobot/
flake8 knobot/
```

### Testing

Run tests using pytest:
```bash
pytest
```

## Project Structure

```
knobot/
├── knobot/
│   ├── __init__.py
│   ├── api.py
│   ├── agent.py
│   ├── config.py
│   ├── logger.py
│   ├── rag.py
│   ├── train.py
│   └── inference.py
├── tests/
├── models/
├── requirements.txt
├── setup_venv.sh
└── README.md
```

## Model Training Configuration

The training process can be configured using the `TrainingConfig` class in `knobot/train.py`. Key parameters include:

- `model_name`: Base model to fine-tune (default: "google/flan-t5-small")
- `num_train_epochs`: Number of training epochs (default: 10)
- `per_device_train_batch_size`: Batch size per device (default: 3)
- `max_length`: Maximum sequence length (default: 512)

## Model Inference Configuration

The inference process can be configured using the `InferenceConfig` class in `knobot/inference.py`. Key parameters include:

- `model_path`: Path to the trained model (default: "./models")
- `max_length`: Maximum sequence length (default: 512)
- `num_beams`: Number of beams for beam search (default: 4)
- `early_stopping`: Whether to use early stopping (default: True)

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 