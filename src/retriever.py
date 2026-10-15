import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NoteRetriever:
    """
    Retrieves candidate context notes for a given route and week.
    Uses TF-IDF vectorization and cosine similarity over note texts and metadata.
    100% deterministic and runs locally with zero external API dependencies.
    """
    def __init__(self, notes_df: pd.DataFrame):
        self.notes_df = notes_df.copy()
        # Build search documents combining note_id, date, applies_to, and note text
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
            
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
        
    def retrieve_candidates(self, route: str, week_of: pd.Timestamp, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidate notes for a given route and week.
        """
        query = (
            f"{route} "
            f"{week_of.strftime('%Y-%m-%d')} "
            f"{week_of.strftime('%B %Y')} "
            f"freight transport cost"
        )
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Rank by score descending
        ranked_indices = np.argsort(scores)[::-1]
        
        candidates = []
        for idx in ranked_indices[:top_k]:
            row = self.notes_df.iloc[idx]
            candidates.append({
                'note_id': row['note_id'],
                'date': row['date'],
                'applies_to': row['applies_to'],
                'note': row['note'],
                'score': float(scores[idx])
            })
            
        return candidates
