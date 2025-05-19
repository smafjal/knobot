import uvicorn
from knobot.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "knobot.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    ) 