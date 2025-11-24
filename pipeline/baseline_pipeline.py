from chunking.simple_chunker import SimpleChunker
from retriever.copali_retriever import Retriever
from retriever.rrf_text_retriever import HybridRetriever
from parse.pdf_parser import PDFParser
from parse.pymupdf_parser import PyMuPDFParser
from chat import BaseLLM, GeminiChat
from typing import List
from langchain_core.documents import Document
import os
from prompt.baseline_prompt import generation_prompt
import argparse
from utils.profile import latency_context, export_latency


class SimpleRAGPipeline:
    def __init__(
        self,
        vision_embedding_dir: str,
        doc_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        vision_topk: int,
        keyword_k: int,
        dense_k: int,
        final_k: int,
        hybrid_weights: List[float],
        model: str,
        embedding_model: str,
    ):
        self.vision_retriever = Retriever(doc_dir, vision_embedding_dir)
        self.doc_dir = doc_dir
        self.page_parser: PDFParser = PyMuPDFParser()
        self.chunker = SimpleChunker(chunk_size, chunk_overlap)
        self.text_retriever: HybridRetriever = None
        self.llm: BaseLLM = GeminiChat(model=model)
        self.vision_topk = vision_topk
        self.keyword_k = keyword_k
        self.dense_k = dense_k
        self.final_k = final_k
        self.hybrid_weights = hybrid_weights
        self.embedding_model = embedding_model

    def query(self, question: str):
        # stage 1: retrieve by vision embeddings
        pages = []
        with latency_context("VisionRetrieval"):
            pages = self.vision_retriever.retrieve(
                question, self.vision_topk
            )  # every page result is a dict with keys: doc_id, page_num, score
        # stage 2: parse image and do chunking
        page_documents: List[Document] = []
        with latency_context("ParseAndChunk"):
            for page in pages:
                doc_id, page_num = page["doc_id"], page["page_num"]
                doc_path = os.path.join(self.doc_dir, doc_id)
                page_content = self.page_parser.parse_page(doc_path, page_num)
                text = page_content.text
                chunks = self.chunker.split_text(text)
                for chunk in chunks:
                    metadata = {}
                    metadata.update(page_content.metadata)
                    metadata["page_num"] = page_num
                    metadata["doc_id"] = doc_id
                    document = Document(page_content=chunk, metadata=metadata)
                    page_documents.append(document)
        # stage 3: Text Retrieval
        with latency_context("TextRetrieval"):
            self.text_retriever = HybridRetriever(
                self.keyword_k, self.dense_k, page_documents, self.embedding_model
            )
            retrieved_documents = self.text_retriever.retrieve(
                question, self.final_k, self.hybrid_weights
            )
            context = self._create_stuff_context(retrieved_documents)
        # stage 4: generation
        with latency_context("FinalGeneration"):
            prompt = generation_prompt.format(context=context, question=question)
            answer = self.llm.chat(prompt)
        return answer

    def _create_stuff_context(self, documents: List[Document]) -> str:
        if documents:
            texts = [d.page_content for d in documents]
            return "\n".join(texts)
        else:
            return "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--doc_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--vision_k", type=int, default=10)
    parser.add_argument("--keyword_k", type=int, default=20)
    parser.add_argument("--dense_k", type=int, default=20)
    parser.add_argument("--final_k", type=int, default=5)
    parser.add_argument("--keyword_weight", type=float, default=0.5)
    parser.add_argument("--generation_model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--metrics_output_dir", type=str, default="output")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model: 'text-embedding-3-small', 'text-embedding-3-large', 'BAAI/bge-large-en-v1.5', etc.",
    )
    args = parser.parse_args()

    rag = SimpleRAGPipeline(
        args.embedding_dir,
        args.doc_dir,
        args.chunk_size,
        args.chunk_overlap,
        args.vision_k,
        args.keyword_k,
        args.dense_k,
        args.final_k,
        [args.keyword_weight, 1 - args.keyword_weight],
        args.generation_model,
        args.embedding_model,
    )

    while True:
        query = input("Enter Query (enter /exit to quit):")
        if query == "/exit":
            break
        else:
            answer = rag.query(query)
            print(answer)
    os.makedirs(args.metrics_output_dir, exist_ok=True)
    output_path = os.path.join(args.metrics_output_dir, "metrics.csv")
    export_latency(output_path, format="csv")


if __name__ == "__main__":
    main()
