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

from chat import BaseLLM, GeminiChat
from chunking.simple_chunker import SimpleChunker
from parse.pymupdf_parser import PyMuPDFParser
from prompt.baseline_prompt import generation_prompt
from utils.profile import export_latency, latency_context


def _create_embeddings(model_name: str):
    """Factory for OpenAI or HuggingFace embeddings."""
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
        return OpenAIEmbeddings(model=model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


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
    ):
        self.doc_dir = doc_dir
        self.page_parser = PyMuPDFParser()
        self.chunker = SimpleChunker(chunk_size, chunk_overlap)
        self.llm: BaseLLM = GeminiChat(model=llm_model)
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.doc_filter = set(doc_filter) if doc_filter else None
        self.single_doc_mode = single_doc_mode

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
        pdf_paths = glob.glob(os.path.join(self.doc_dir, "**/*.pdf"), recursive=True)
        if not pdf_paths:
            raise ValueError(f"No PDFs found under {self.doc_dir}")

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
                # In single_doc_mode we only expect/process one document; break after first.
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
    parser.add_argument("--metrics_output_dir", type=str, default="output")

    args = parser.parse_args()

    pipeline = TextRAGPipeline(
        doc_dir=args.doc_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        llm_model=args.generation_model,
        embedding_model=args.embedding_model,
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
