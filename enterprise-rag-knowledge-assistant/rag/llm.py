from __future__ import annotations
import os

def generate_answer(question: str, context: str):
    key=os.getenv('OPENAI_API_KEY')
    if not key:
        return None, 'extractive'
    try:
        from openai import OpenAI
        client=OpenAI(api_key=key)
        response=client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),temperature=0,
            messages=[{'role':'system','content':'Answer only from the supplied context. If context is insufficient, say so. Keep the answer concise.'},
                      {'role':'user','content':f'Context:\n{context}\n\nQuestion: {question}'}])
        return response.choices[0].message.content, 'llm'
    except Exception:
        return None, 'extractive'
