import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_chunks = {}
        self.child_to_parent_map = {}
    
    def add_documents(self, hierarchical_data: Dict):
        self.parent_chunks = hierarchical_data['parent_chunks']
        self.child_to_parent_map = hierarchical_data['child_to_parent_map']
        child_chunks = hierarchical_data['child_chunks']
        
        if not child_chunks:
            return
        
        ids = [chunk['id'] for chunk in child_chunks]
        texts = [chunk['text'] for chunk in child_chunks]
        metadatas = [{'parent_id': chunk['parent_id'], 'filename': chunk['filename']} for chunk in child_chunks]
        
        embeddings = self.embedding_model.encode(texts).tolist()
        
        existing_ids = set(self.collection.get()['ids'])
        new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]
        
        if new_indices:
            self.collection.add(
                ids=[ids[i] for i in new_indices],
                embeddings=[embeddings[i] for i in new_indices],
                documents=[texts[i] for i in new_indices],
                metadatas=[metadatas[i] for i in new_indices]
            )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved_results = []
        seen_parents = set()
        
        for i in range(len(results['ids'][0])):
            child_id = results['ids'][0][i]
            parent_id = results['metadatas'][0][i]['parent_id']
            
            if parent_id not in seen_parents and parent_id in self.parent_chunks:
                seen_parents.add(parent_id)
                retrieved_results.append({
                    'child_text': results['documents'][0][i],
                    'parent_text': self.parent_chunks[parent_id]['text'],
                    'filename': results['metadatas'][0][i]['filename'],
                    'distance': results['distances'][0][i],
                    'relevance_score': 1 - results['distances'][0][i]
                })
        
        return retrieved_results
    
    def clear(self):
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_chunks = {}
        self.child_to_parent_map = {}
