from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
from collections import defaultdict

def rrf_fuse(doc_lists, weights=None, c=60) -> List[Document]:
    if weights is None:
        weights = [1/len(doc_lists)] * len(doc_lists)
    scores = defaultdict(float)
    metas  = {}

    for w, docs in zip(weights, doc_lists):
        for r, d in enumerate(docs, start=1):
            key = d.page_content
            scores[key] += w / (c + r)
            metas[key] = d.metadata

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [type(docs[0])(page_content=k, metadata=metas[k]) for k, _ in fused]

class HybridRetriever:
    def __init__(self, keyword_k: int, dense_k: int, documents: List[Document]):
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = keyword_k
        vs = FAISS.from_documents(documents, OpenAIEmbeddings())
        self.dense = vs.as_retriever(search_kwargs={"k": dense_k})
    def retrieve(self, question: str, top_k: int, weights: List[float] = None) -> List[Document]:
        bm25_docs = self.bm25.invoke(question)
        dense_docs = self.dense.invoke(question)
        results = rrf_fuse([bm25_docs, dense_docs], weights)
        return results[:top_k]
