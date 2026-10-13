from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    def __init__(self, knowledge_dir: Path, chunk_words: int = 95, overlap: int = 18):
        self.knowledge_dir=knowledge_dir; self.chunk_words=chunk_words; self.overlap=overlap
        self.chunks=[]; self.vectorizer=None; self.matrix=None

    def _chunk(self, text, source):
        # Keep paragraph boundaries where possible, then create overlapping windows.
        clean=re.sub(r'\s+',' ',text).strip(); words=clean.split(); out=[]; i=0; cid=0
        while i < len(words):
            piece=' '.join(words[i:i+self.chunk_words]);
            if piece: out.append({'source':source,'chunk_id':cid,'text':piece})
            cid+=1; i += max(1,self.chunk_words-self.overlap)
        return out

    def index(self):
        chunks=[]
        for p in sorted(self.knowledge_dir.glob('*.md')):
            chunks.extend(self._chunk(p.read_text(encoding='utf-8'),p.name))
        self.chunks=chunks
        self.vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True)
        self.matrix=self.vectorizer.fit_transform([c['text'] for c in chunks])

    def retrieve(self, question: str, top_k: int = 3):
        if self.matrix is None: self.index()
        q=self.vectorizer.transform([question]); scores=cosine_similarity(q,self.matrix).ravel(); idx=np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i],'score':round(float(scores[i]),4)} for i in idx]

    def extractive_answer(self, question, results):
        if not results: return 'I could not find relevant information in the indexed knowledge base.'
        qterms=set(re.findall(r'[a-z0-9]+',question.lower()))
        sentences=[]
        for r in results:
            for s in re.split(r'(?<=[.!?])\s+',r['text']):
                sterms=set(re.findall(r'[a-z0-9]+',s.lower())); score=len(qterms & sterms)/(len(qterms)+1e-6)
                sentences.append((score,s))
        best=[s for score,s in sorted(sentences,reverse=True) if score>0][:2]
        return ' '.join(best) if best else results[0]['text']
