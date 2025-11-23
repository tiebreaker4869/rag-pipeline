generation_prompt: str = """You are a helpful assistant. Answer the question based on the given context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the information from the context above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and accurate
- Cite specific parts of the context when possible

Answer:"""
