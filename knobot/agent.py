from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from knobot.inference import ModelInference, InferenceConfig
from knobot.rag import RAGSystem
from knobot.logger import setup_logger
from knobot.config import settings

logger = setup_logger(__name__)

class Question(BaseModel):
    text: str = Field(..., description="The question text")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the question")

class AgentError(Exception):
    pass

class ModelLoadError(AgentError):
    pass

class RAGError(AgentError):
    pass

class Agent:
    def __init__(self, model_path: Optional[Path] = None, rag_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.rag_path = rag_path or settings.RAG_PATH
        
        # Initialize inference and RAG
        self.inference = ModelInference(InferenceConfig(model_path=str(self.model_path)))
        self.inference.load_model()
        self.rag = RAGSystem()
        
        try:
            logger.info(f"Loading RAG system from {self.rag_path}")
            self.rag.load(self.rag_path)
        except Exception as e:
            logger.warning(f"No existing RAG system found: {str(e)}")
    
    def add_documents(self, documents: List[str]) -> None:
        try:
            logger.info(f"Adding {len(documents)} documents to RAG system")
            self.rag.add_documents(documents)
            self.rag.save(self.rag_path)
            logger.info("Documents added successfully")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise RAGError(f"Failed to add documents: {str(e)}")
    
    def process_question(self, question: Question) -> str:
        try:
            logger.info(f"Processing question: {question.text}")
            relevant_docs = self.rag.query(question.text)
            context = "\n".join(relevant_docs)
            
            # Format the question with context
            augmented_question = f"Context: {context}\n\nQuestion: {question.text}"
            logger.info(f"Augmented question: {augmented_question}")
            
            # Generate response using the model
            response = self.inference.generate_response(augmented_question)
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process question: {str(e)}")
            raise AgentError(f"Failed to process question: {str(e)}") 