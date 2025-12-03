"""Vision-based retrieval using ColPali (ColQwen2.5) multi-vector embeddings."""

import os
from glob import glob
from typing import List, Dict, Optional

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


# Global cache for ColPali model to avoid reloading
_MODEL_CACHE = {}


class ColPaliRetriever:
    """Vision-based retrieval using pre-computed ColPali embeddings.

    Assumes a single document directory containing .pt, .pdf, and .md files.
    """

    def __init__(
        self,
        doc_dir: str,
        top_k: int = 5,
    ):
        """Initialize ColPali retriever.

        Args:
            doc_dir: Directory containing .pt embedding file, PDF, and markdown files
            top_k: Default number of top pages to retrieve
        """
        self.doc_dir = doc_dir
        self.top_k = top_k

        # Find the single .pt file in the directory
        pt_files = glob(os.path.join(doc_dir, "*.pt"))
        if not pt_files:
            raise ValueError(f"No .pt embedding file found in {doc_dir}")
        if len(pt_files) > 1:
            print(f"Warning: Multiple .pt files found, using the first one: {pt_files[0]}")

        self.embedding_path = pt_files[0]

        # Setup model with caching
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = "colqwen2.5-v0.2"

        if cache_key in _MODEL_CACHE:
            print(f"[INFO] Reusing cached ColQwen2.5 model")
            self.processor, self.model = _MODEL_CACHE[cache_key]
        else:
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
            _MODEL_CACHE[cache_key] = (self.processor, self.model)
            print(f"[INFO] ColQwen2.5 model loaded")

        print(f"[INFO] Using embeddings from: {self.embedding_path}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, any]]:
        """Retrieve top-k most relevant pages from the document.

        Args:
            query: Query string
            top_k: Number of top pages to return (defaults to self.top_k)

        Returns:
            List of dictionaries with page_num and score (sorted by score desc)
        """
        k = top_k or self.top_k

        # Encode query
        print(f"[INFO] Processing query: {query}")
        batch_queries = self.processor.process_queries([query]).to(self.device)

        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)

        # Load embedding for this document
        try:
            doc_embeddings = torch.load(self.embedding_path, map_location=self.device)

            # Multi-vector similarity scoring
            with torch.no_grad():
                scores = self.processor.score_multi_vector(
                    query_embeddings, doc_embeddings
                )

            # Extract individual page scores
            page_scores = (
                scores.squeeze(0).tolist() if scores.dim() > 1 else [scores.item()]
            )

            # Create results list
            all_page_scores = [
                {"page_num": page_num, "score": score}
                for page_num, score in enumerate(page_scores, start=1)
            ]

            # Free memory
            del doc_embeddings, scores
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Failed to process {self.embedding_path}: {e}")
            raise

        # Sort by score (descending) and get top-k
        all_page_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_page_scores[:k]

        print(f"\n[INFO] Top {k} pages:")
        for i, result in enumerate(top_results):
            print(f"  {i+1}. Page {result['page_num']}: {result['score']:.4f}")

        return top_results
