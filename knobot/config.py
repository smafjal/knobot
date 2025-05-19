from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8991
    API_TITLE: str = "Knobot Question Answering API"
    API_VERSION: str = "0.1.0"
    
    # Model Settings
    MODEL_PATH: Path = Path("./models")
    RAG_PATH: Path = Path("./data/rag")
    
    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()