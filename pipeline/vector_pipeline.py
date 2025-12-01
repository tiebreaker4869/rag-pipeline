import argparse
import glob
import os
from typing import List, Optional

import fitz
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from chat import BaseLLM, GeminiChat, OpenAIChat
from chunking.simple_chunker import SimpleChunker
from parse.pymupdf_parser import PyMuPDFParser
from prompt.baseline_prompt import generation_prompt
from rerank import BaseReranker, BGEReranker
from utils.profile import export_latency, latency_context


# Global cache for embedding models to prevent memory leaks
_EMBEDDING_CACHE = {}


def _create_embeddings(model_name: str):
    """Factory for OpenAI or HuggingFace embeddings with caching."""
    # Check cache first
    if model_name in _EMBEDDING_CACHE:
        print(f"[INFO] Reusing cached embedding model: {model_name}")
        return _EMBEDDING_CACHE[model_name]

    openai_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    if model_name in openai_models:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                f"OPENAI_API_KEY environment variable is required for model {model_name}"
            )
        embeddings = OpenAIEmbeddings(model=model_name)
    else:
        print(f"[INFO] Loading embedding model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Cache the model
    _EMBEDDING_CACHE[model_name] = embeddings
    return embeddings


class TextRAGPipeline:
    """
    Text-only RAG baseline: parse PDFs -> chunk -> embed -> FAISS retrieval -> generation.
    """

    def __init__(
        self,
        doc_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        top_k: int = 5,
        llm_model: str = "gemini-1.5-flash",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        doc_filter: Optional[List[str]] = None,
        single_doc_mode: bool = False,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-base",
        rerank_top_k: Optional[int] = None,
    ):
        self.doc_dir = doc_dir
        self.page_parser = PyMuPDFParser()
        self.chunker = SimpleChunker(chunk_size, chunk_overlap)
        self.llm: BaseLLM = GeminiChat(model=llm_model)
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.doc_filter = set(doc_filter) if doc_filter else None
        self.single_doc_mode = single_doc_mode
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else top_k

        # Initialize reranker if enabled
        self.reranker: Optional[BaseReranker] = None
        if use_reranker:
            self.reranker = BGEReranker(model_name=reranker_model)

        with latency_context("BuildIndex"):
            self.retriever, self.vectorstore = self._build_vector_index()

    def _build_vector_index(self):
        with latency_context("Chunking"):
            documents = self._load_documents()
        embeddings = _create_embeddings(self.embedding_model)
        vs = FAISS.from_documents(documents, embeddings)
        retriever = vs.as_retriever(search_kwargs={"k": self.top_k})
        return retriever, vs

    def _load_documents(self) -> List[Document]:
        # First try to find .txt files (OCR preprocessed)
        txt_paths = glob.glob(os.path.join(self.doc_dir, "**/*.txt"), recursive=True)

        # Fallback to PDFs if no txt files found
        if not txt_paths:
            pdf_paths = glob.glob(os.path.join(self.doc_dir, "**/*.pdf"), recursive=True)
            if not pdf_paths:
                raise ValueError(f"No TXT or PDF files found under {self.doc_dir}")
            return self._load_from_pdfs(pdf_paths)

        return self._load_from_txt_files(txt_paths)

    def _load_from_txt_files(self, txt_paths: List[str]) -> List[Document]:
        """Load documents from OCR-preprocessed .txt files"""
        docs: List[Document] = []
        for txt_path in txt_paths:
            # Extract doc_id from filename (e.g., "docname.txt" -> "docname.pdf")
            doc_name = os.path.basename(txt_path)
            doc_id = doc_name.replace(".txt", ".pdf")

            # Filter by doc_id if specified
            if self.doc_filter and doc_id not in self.doc_filter:
                continue

            # Read the entire text file
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Warning: Failed to read {txt_path}: {e}")
                continue

            # Skip empty files
            if not text.strip():
                print(f"Warning: {doc_id} has no text content, skipping...")
                continue

            # Chunk the entire document text
            chunks = self.chunker.split_text(text)
            if not chunks:
                print(f"Warning: {doc_id} produced no chunks after splitting, skipping...")
                continue

            for chunk_idx, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "source": txt_path,
                }
                docs.append(Document(page_content=chunk, metadata=metadata))

            if self.single_doc_mode:
                break

        return docs

    def _load_from_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Load documents from PDF files (fallback method)"""
        docs: List[Document] = []
        for pdf_path in pdf_paths:
            doc_id = os.path.basename(pdf_path)
            if self.doc_filter and doc_id not in self.doc_filter:
                continue
            pdf = fitz.open(pdf_path)
            total_pages = pdf.page_count
            pdf.close()

            for page_num in range(1, total_pages + 1):
                page_content = self.page_parser.parse_page(pdf_path, page_num)
                for chunk in self.chunker.split_text(page_content.text):
                    metadata = dict(page_content.metadata or {})
                    metadata.update({"doc_id": doc_id, "page_num": page_num})
                    docs.append(Document(page_content=chunk, metadata=metadata))

            if self.single_doc_mode:
                break

        return docs

    def query(self, question: str, doc_id: Optional[str] = None) -> str:
        with latency_context("TextRetrieval"):
            # If a doc_id is provided, filter results to that document's metadata.
            if doc_id:
                retrieved_documents = self.vectorstore.similarity_search(
                    question, k=self.top_k, filter={"doc_id": doc_id}
                )
            else:
                retrieved_documents = self.retriever.invoke(question)

        # Apply reranking if enabled
        if self.use_reranker and self.reranker:
            with latency_context("Rerank"):
                retrieved_documents = self.reranker.rerank(
                    question, retrieved_documents, top_k=self.rerank_top_k
                )

        with latency_context("FinalGeneration"):
            context = self._create_context(retrieved_documents)
            prompt = generation_prompt.format(context=context, question=question)
            answer = self.llm.chat(prompt)
        return answer

    def _create_context(self, documents: List[Document]) -> str:
        if documents:
            return "\n".join([d.page_content for d in documents])
        return "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Text-only RAG baseline with FAISS vector search."
    )
    parser.add_argument("--doc_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--generation_model", type=str, default="gemini-1.5-flash"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="OpenAI embeddings (text-embedding-3-*) or HuggingFace model name.",
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Enable reranking with BGE cross-encoder",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Reranker model name (BAAI/bge-reranker-base or BAAI/bge-reranker-large)",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=None,
        help="Number of documents after reranking (defaults to same as top_k)",
    )
    parser.add_argument("--metrics_output_dir", type=str, default="output")

    args = parser.parse_args()

    pipeline = TextRAGPipeline(
        doc_dir=args.doc_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        llm_model=args.generation_model,
        embedding_model=args.embedding_model,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        rerank_top_k=args.rerank_top_k,
    )

    while True:
        query = input("Enter Query (enter /exit to quit):")
        if query == "/exit":
            break
        answer = pipeline.query(query)
        print(answer)

    os.makedirs(args.metrics_output_dir, exist_ok=True)
    output_path = os.path.join(args.metrics_output_dir, "vector_metrics.csv")
    export_latency(output_path, format="csv")


if __name__ == "__main__":
    main()
