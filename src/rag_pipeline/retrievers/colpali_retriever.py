"""Vision-based retrieval using ColPali (ColQwen2.5) multi-vector embeddings."""

import os
from glob import glob
from typing import List, Dict, Optional

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ColPaliRetriever:
    """Vision-based retrieval using pre-computed ColPali embeddings."""

    def __init__(
        self,
        doc_dir: str,
        embedding_dir: Optional[str] = None,
        top_k: int = 5,
    ):
        """Initialize ColPali retriever.

        Args:
            doc_dir: Directory containing PDF documents
            embedding_dir: Directory containing .pt embedding files
                          (defaults to doc_dir with 'documents' -> 'embeddings')
            top_k: Default number of top pages to retrieve
        """
        self.doc_dir = doc_dir
        self.embedding_dir = embedding_dir or doc_dir.replace("documents", "embeddings")
        self.top_k = top_k

        # Setup model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading ColQwen2.5 model on {self.device}...")

        self.processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        self.model = (
            ColQwen2_5.from_pretrained(
                "vidore/colqwen2.5-v0.2",
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )

        # Index embedding files: doc_id -> embedding_path
        self.embedding_map = {}
        embedding_files = glob(
            os.path.join(self.embedding_dir, "**/*.pt"), recursive=True
        )
        for emb_path in embedding_files:
            doc_id = os.path.basename(emb_path).replace(".pt", ".pdf")
            self.embedding_map[doc_id] = emb_path

        print(f"[INFO] Indexed {len(self.embedding_map)} documents")

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, any]]:
        """Retrieve top-k most relevant pages.

        Args:
            query: Query string
            doc_id: Optional document ID to restrict search to specific document.
                   If None, searches across all documents.
            top_k: Number of top pages to return (defaults to self.top_k)

        Returns:
            List of dictionaries with doc_id, page_num, and score (sorted by score desc)
        """
        k = top_k or self.top_k

        # Encode query once
        print(f"[INFO] Processing query: {query}")
        batch_queries = self.processor.process_queries([query]).to(self.device)

        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)

        # Determine which documents to search
        if doc_id is not None:
            # Search only in specified document
            if doc_id not in self.embedding_map:
                raise ValueError(f"Document {doc_id} not found in index")
            search_docs = {doc_id: self.embedding_map[doc_id]}
            print(f"[INFO] Searching in document: {doc_id}")
        else:
            # Search all documents
            search_docs = self.embedding_map
            print(f"[INFO] Searching across {len(search_docs)} documents")

        # Collect scores for pages
        all_page_scores = []

        for doc_id, emb_path in search_docs.items():
            try:
                # Load embedding for this document
                doc_embeddings = torch.load(emb_path, map_location=self.device)

                # Multi-vector similarity scoring
                with torch.no_grad():
                    scores = self.processor.score_multi_vector(
                        query_embeddings, doc_embeddings
                    )

                # Extract individual page scores
                page_scores = (
                    scores.squeeze(0).tolist() if scores.dim() > 1 else [scores.item()]
                )

                # Add each page's score to the list
                for page_num, score in enumerate(page_scores, start=1):
                    all_page_scores.append({
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "score": score,
                    })

                # Free memory
                del doc_embeddings, scores
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] Failed to process {emb_path}: {e}")
                continue

        # Sort all pages by score (descending) and get top-k
        all_page_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_page_scores[:k]

        print(f"\n[INFO] Top {k} pages:")
        for i, result in enumerate(top_results):
            print(
                f"  {i+1}. {result['doc_id']} - Page {result['page_num']}: "
                f"{result['score']:.4f}"
            )

        return top_results
