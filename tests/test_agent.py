import pytest
from knobot.agent import Agent, Question, AgentError

@pytest.fixture
def agent():
    return Agent()

def test_question_creation():
    question = Question(text="Test question")
    assert question.text == "Test question"
    assert question.context is None

def test_add_documents(agent):
    documents = ["Test document 1", "Test document 2"]
    agent.add_documents(documents)
    # Note: We can't easily test the internal state of the RAG system
    # This is more of an integration test

def test_process_question(agent):
    question = Question(text="What is the capital of France?")
    response = agent.process_question(question)
    assert isinstance(response, str)
    assert len(response) > 0

def test_invalid_question(agent):
    with pytest.raises(AgentError):
        agent.process_question(Question(text=""))