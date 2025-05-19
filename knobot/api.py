from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from knobot.agent import Agent, AgentError, Question
from knobot.config import settings
from knobot.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="A Question Answering API powered by RAG and local language models"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = Agent()

class QuestionRequest(BaseModel):
    text: str = Field(..., description="The question text to process")
    context: Optional[List[str]] = None

class DocumentRequest(BaseModel):
    documents: List[str] = Field(..., description="List of documents to add to the RAG system")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")

@app.post("/ask", response_model=dict, responses={500: {"model": ErrorResponse}})
async def ask_question(request: QuestionRequest):
    """
    Process a question and return a response.
    
    Args:
        request: QuestionRequest containing the question text
        
    Returns:
        dict: Response containing the generated answer
        
    Raises:
        HTTPException: If question processing fails
    """
    try:
        logger.info(f"Received question: {request.text}")
        question = Question(text=request.text, context=request.context)
        response = agent.process_question(question)
        return {"response": response}
    except AgentError as e:
        logger.error(f"Agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/documents", response_model=dict, responses={500: {"model": ErrorResponse}})
async def add_documents(request: DocumentRequest):
    """
    Add documents to the RAG system.
    
    Args:
        request: DocumentRequest containing the documents to add
        
    Returns:
        dict: Success message with number of documents added
        
    Raises:
        HTTPException: If document addition fails
    """
    try:
        logger.info(f"Adding {len(request.documents)} documents")
        agent.add_documents(request.documents)
        return {"message": f"Added {len(request.documents)} documents successfully"}
    except AgentError as e:
        logger.error(f"Agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

