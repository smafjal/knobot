from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

class RAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        
    def add_documents(self, documents: List[str]):
        # Encode documents
        embeddings = self.model.encode(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store original documents
        self.documents = documents
        
    def query(self, query: str, k: int = 3) -> List[str]:
        if not self.index:
            return []
            
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant documents
        return [self.documents[i] for i in indices[0]]
    
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save documents
        with open(os.path.join(directory, "documents.json"), "w") as f:
            json.dump(self.documents, f)
    
    def load(self, directory: str):
        # Load index
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load documents
        with open(os.path.join(directory, "documents.json"), "r") as f:
            self.documents = json.load(f) 