import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class FAISSDenseRetriever:
    """
    Dense semantic embedding retriever backed by a FAISS index.
    Embeds notes into a dense semantic vector space and builds a FAISS IndexFlatIP.
    Runs 100% locally and deterministically.
    """
    def __init__(self, notes_df: pd.DataFrame, embedding_dim: int = 16):
        if not FAISS_AVAILABLE or faiss is None:
            raise ImportError(
                "faiss is not installed. To use FAISSDenseRetriever, install faiss-cpu. "
                "The core pipeline runs deterministically without faiss."
            )
        self.notes_df = notes_df.copy()
        self.embedding_dim = min(embedding_dim, len(notes_df) - 1)
        
        self.docs = []
        for _, row in self.notes_df.iterrows():
            doc = (
                f"{row['note_id']} "
                f"{row['applies_to']} "
                f"{row['date'].strftime('%Y-%m-%d')} "
                f"{row['date'].strftime('%B %Y')} "
                f"{row['note']}"
            )
            self.docs.append(doc)
            
        # Dense Semantic Projector (LSA / SVD dense embeddings)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_mat = self.vectorizer.fit_transform(self.docs)
        
        self.svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        dense_vecs = self.svd.fit_transform(tfidf_mat)
        dense_vecs = normalize(dense_vecs, norm='l2', axis=1).astype(np.float32)
        
        # Build FAISS Index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(dense_vecs)

    def retrieve_candidates(self, route: str, week_of: pd.Timestamp, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the query into dense vector space and queries the FAISS index.
        """
        query = (
            f"{route} "
            f"{week_of.strftime('%Y-%m-%d')} "
            f"{week_of.strftime('%B %Y')} "
            f"freight transport cost"
        )
        q_tfidf = self.vectorizer.transform([query])
        q_dense = self.svd.transform(q_tfidf)
        q_dense = normalize(q_dense, norm='l2', axis=1).astype(np.float32)
        
        top_k = min(top_k, len(self.notes_df))
        scores, indices = self.index.search(q_dense, top_k)
        
        candidates = []
        for rank in range(top_k):
            idx = indices[0][rank]
            score = float(scores[0][rank])
            row = self.notes_df.iloc[idx]
            candidates.append({
                'note_id': row['note_id'],
                'date': row['date'],
                'applies_to': row['applies_to'],
                'note': row['note'],
                'score': score
            })
        return candidates
