"""RAG prompt template adapted for single-iteration retrieval."""

rag_generation_prompt: str = """You are an AI assistant capable of analyzing documents and extracting relevant information to answer questions.

Consider this question about the document:
<question>
{question}
</question>

The following pages have been retrieved as potentially relevant to your question:
<retrieved_pages>
{page_numbers}
</retrieved_pages>

Raw text extracted from the retrieved pages:
<page_text>
{context}
</page_text>

IMPORTANT: Carefully analyze the provided text to understand the document content and answer the question accurately.

<scratchpad>
1. List key elements from the text that relate to the question
2. Identify specific details that directly answer the question
3. Make connections between the document information and the question
4. Determine if the provided information is sufficient to answer the question
5. Consider whether the answer might require information not present in the retrieved pages
</scratchpad>

Based on your analysis, provide your response:

If the provided pages contain sufficient information to answer the question:
- Provide a clear and concise answer that directly addresses the question
- Include an explanation of how you arrived at this conclusion using information from the document
- Cite specific page numbers when referencing information

If the document does not contain the information needed to answer the question:
- Clearly state that the information is not available in the provided context
- Explain what information is missing

Your response:"""


# Simple baseline prompt for basic RAG
baseline_prompt: str = """You are a helpful assistant. Answer the question based on the given context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the information from the context above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and accurate
- Cite specific parts of the context when possible

Answer:"""
