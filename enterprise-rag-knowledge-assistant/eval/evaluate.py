from pathlib import Path
from rag.engine import RAGEngine

QUESTIONS=[
    ('How long are audit logs retained?','security_policy.md'),
    ('What can I export from analytics?','product_guide.md'),
    ('When are enterprise invoices due?','billing_faq.md'),
]
def evaluate(k=3):
    e=RAGEngine(Path(__file__).resolve().parents[1]/'knowledge'); e.index()
    hits=0; rr=0.0
    for q,expected in QUESTIONS:
        r=e.retrieve(q,k); ranks=[i+1 for i,x in enumerate(r) if x['source']==expected]
        if ranks: hits+=1; rr += 1/min(ranks)
    metrics={'questions':len(QUESTIONS),f'hit@{k}':round(hits/len(QUESTIONS),4),'mrr':round(rr/len(QUESTIONS),4)}
    print(metrics); return metrics
if __name__=='__main__': evaluate()
