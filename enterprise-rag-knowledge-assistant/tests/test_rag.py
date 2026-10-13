from pathlib import Path
from fastapi.testclient import TestClient
from rag.engine import RAGEngine
from app.main import app

def test_security_retrieval():
    e=RAGEngine(Path(__file__).resolve().parents[1]/'knowledge'); e.index()
    assert e.retrieve('How long are audit records retained?',1)[0]['source']=='security_policy.md'

def test_ask_endpoint():
    r=TestClient(app).post('/ask',json={'question':'When are enterprise invoices due?','top_k':3})
    assert r.status_code==200
    assert r.json()['citations']
