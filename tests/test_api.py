import pytest
from fastapi.testclient import TestClient
from knobot.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ask_question():
    response = client.post(
        "/ask",
        json={"text": "What is the capital of France?"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_add_documents():
    response = client.post(
        "/documents",
        json={"documents": ["Paris is the capital of France."]}
    )
    assert response.status_code == 200
    assert "message" in response.json()
    assert "successfully" in response.json()["message"]