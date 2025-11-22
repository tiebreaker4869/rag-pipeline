import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import os
from glob import glob
from typing import List, Dict

class Retriever:
    def __init__(self, doc_dir: str, embedding_dir: str = None, top_k: int = 5):
        self.doc_dir = doc_dir
        self.embedding_dir = embedding_dir or doc_dir.replace("documents", "embeddings")
        self.top_k = top_k

        # Set up embedding-based retrieval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor_emb = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
        self.model_emb = ColQwen2_5.from_pretrained(
                "vidore/colqwen2.5-v0.2", torch_dtype=torch.bfloat16
            ).to(self.device).eval()

        # Get list of all embedding files
        self.embedding_files = glob(os.path.join(self.embedding_dir, "**/*.pt"), recursive=True)
        print(f"[INFO] Found {len(self.embedding_files)} embedding files in {self.embedding_dir}")

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, any]]:
        """
        Retrieve top-k most relevant pages across all documents for the given question
        Loads embeddings one by one to avoid memory issues

        Args:
            question: Query string
            top_k: Number of top pages to return (default: self.top_k)

        Returns:
            List of dictionaries containing doc_id, page_num, and score, sorted by score (descending)
        """
        if top_k is None:
            top_k = self.top_k

        # Encode query once
        print(f"[INFO] Processing query: {question}")
        batch_queries = self.processor_emb.process_queries([question]).to(self.device)

        with torch.no_grad():
            query_embeddings = self.model_emb(**batch_queries)

        # Collect scores for all pages across all documents
        all_page_scores = []

        for emb_path in self.embedding_files:
            doc_id = os.path.basename(emb_path).replace('.pt', '.pdf')

            try:
                # Load embedding for this document
                doc_embeddings = torch.load(emb_path, map_location=self.device)

                # Multi-vector similarity scoring
                with torch.no_grad():
                    scores = self.processor_emb.score_multi_vector(query_embeddings, doc_embeddings)

                # Extract individual page scores
                page_scores = scores.squeeze(0).tolist() if scores.dim() > 1 else [scores.item()]

                # Add each page's score to the list
                for page_num, score in enumerate(page_scores, start=1):
                    all_page_scores.append({
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "score": score
                    })

                # Free memory
                del doc_embeddings, scores
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] Failed to process {emb_path}: {e}")
                continue

        # Sort all pages by score (descending) and get top-k
        all_page_scores.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_page_scores[:top_k]

        print(f"\n[INFO] Top {top_k} pages across all documents:")
        for i, result in enumerate(top_results):
            print(f"  {i+1}. {result['doc_id']} - Page {result['page_num']}: {result['score']:.4f}")

        return top_results
        