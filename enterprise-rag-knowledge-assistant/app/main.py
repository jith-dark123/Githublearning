from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rag.engine import RAGEngine
from rag.llm import generate_answer

ROOT=Path(__file__).resolve().parents[1]; FRONTEND=ROOT/'frontend'; KNOWLEDGE=ROOT/'knowledge'
engine=RAGEngine(KNOWLEDGE); engine.index()
app=FastAPI(title='Enterprise RAG Knowledge Assistant',version='1.0.0')
app.mount('/static',StaticFiles(directory=FRONTEND),name='static')
class AskRequest(BaseModel):
    question:str=Field(min_length=3)
    top_k:int=Field(default=3,ge=1,le=6)
@app.get('/')
def home(): return FileResponse(FRONTEND/'index.html')
@app.get('/health')
def health(): return {'status':'ok','chunks':len(engine.chunks)}
@app.post('/reindex')
def reindex(): engine.index(); return {'status':'indexed','chunks':len(engine.chunks)}
@app.post('/ask')
def ask(req:AskRequest):
    results=engine.retrieve(req.question,req.top_k)
    context='\n\n'.join(r['text'] for r in results)
    answer,mode=generate_answer(req.question,context)
    if mode=='extractive': answer=engine.extractive_answer(req.question,results)
    return {'answer':answer,'mode':mode,'citations':[{'source':r['source'],'chunk':r['chunk_id'],'score':r['score']} for r in results]}
