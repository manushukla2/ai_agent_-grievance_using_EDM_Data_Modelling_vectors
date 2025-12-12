import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import json
import os


class EmpiricalVectorStore:
    def __init__(self, persist_dir: str = "./chroma_empirical_db"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            embedding_function=self.embedding_fn,
            metadata={"description": "Document chunks with hierarchical mapping"}
        )
        
        self.entities_collection = self.client.get_or_create_collection(
            name="entities",
            embedding_function=self.embedding_fn,
            metadata={"description": "Extracted entities with context"}
        )
        
        self.facts_collection = self.client.get_or_create_collection(
            name="facts",
            embedding_function=self.embedding_fn,
            metadata={"description": "Extracted facts and relationships"}
        )
        
        self.parent_chunks: Dict[str, str] = {}
        self.entity_index: Dict[str, List[str]] = {}
        self.fact_index: Dict[str, List[str]] = {}

    def add_chunks(self, chunks: List[Dict], filename: str):
        for chunk in chunks:
            chunk_id = f"{filename}_{chunk['chunk_id']}"
            
            self.chunks_collection.add(
                ids=[chunk_id],
                documents=[chunk['child_text']],
                metadatas=[{
                    'filename': filename,
                    'chunk_id': chunk['chunk_id'],
                    'parent_id': chunk['parent_id'],
                    'chunk_type': 'child'
                }]
            )
            
            self.parent_chunks[chunk_id] = chunk['parent_text']

    def add_entities(self, entities: List[Any], chunk_id: str, filename: str, context: str):
        for i, entity in enumerate(entities):
            entity_id = f"{chunk_id}_entity_{i}"
            
            entity_context = f"{entity.entity_type}: {entity.text} (from: {context[:200]})"
            
            self.entities_collection.add(
                ids=[entity_id],
                documents=[entity_context],
                metadatas=[{
                    'filename': filename,
                    'chunk_id': chunk_id,
                    'entity_type': entity.entity_type,
                    'entity_text': entity.text,
                    'confidence': entity.confidence
                }]
            )
            
            key = f"{entity.entity_type}:{entity.text.lower()}"
            if key not in self.entity_index:
                self.entity_index[key] = []
            self.entity_index[key].append(entity_id)

    def add_facts(self, facts: List[Any], chunk_id: str, filename: str):
        for i, fact in enumerate(facts):
            fact_id = f"{chunk_id}_fact_{i}"
            
            fact_text = f"{fact.subject} {fact.predicate} {fact.object}: {fact.source_text}"
            
            self.facts_collection.add(
                ids=[fact_id],
                documents=[fact_text],
                metadatas=[{
                    'filename': filename,
                    'chunk_id': chunk_id,
                    'subject': fact.subject,
                    'predicate': fact.predicate,
                    'object': fact.object,
                    'confidence': fact.confidence,
                    'source_text': fact.source_text[:500]
                }]
            )
            
            subj_key = fact.subject.lower()
            if subj_key not in self.fact_index:
                self.fact_index[subj_key] = []
            self.fact_index[subj_key].append(fact_id)

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.chunks_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                parent_text = self.parent_chunks.get(chunk_id, results['documents'][0][i])
                distance = results['distances'][0][i] if results['distances'] else 0
                relevance = 1 / (1 + distance)
                
                search_results.append({
                    'chunk_id': chunk_id,
                    'child_text': results['documents'][0][i],
                    'parent_text': parent_text,
                    'filename': results['metadatas'][0][i]['filename'],
                    'relevance_score': relevance
                })
        
        return search_results

    def search_entities(self, query: str, entity_type: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        where_filter = None
        if entity_type:
            where_filter = {"entity_type": entity_type}
        
        results = self.entities_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        entity_results = []
        if results['ids'] and results['ids'][0]:
            for i, entity_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                relevance = 1 / (1 + distance)
                
                entity_results.append({
                    'entity_id': entity_id,
                    'context': results['documents'][0][i],
                    'entity_type': results['metadatas'][0][i]['entity_type'],
                    'entity_text': results['metadatas'][0][i]['entity_text'],
                    'filename': results['metadatas'][0][i]['filename'],
                    'relevance_score': relevance
                })
        
        return entity_results

    def search_facts(self, query: str, top_k: int = 10) -> List[Dict]:
        results = self.facts_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        fact_results = []
        if results['ids'] and results['ids'][0]:
            for i, fact_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                relevance = 1 / (1 + distance)
                
                metadata = results['metadatas'][0][i]
                fact_results.append({
                    'fact_id': fact_id,
                    'fact_text': results['documents'][0][i],
                    'subject': metadata['subject'],
                    'predicate': metadata['predicate'],
                    'object': metadata['object'],
                    'source_text': metadata['source_text'],
                    'filename': metadata['filename'],
                    'relevance_score': relevance
                })
        
        return fact_results

    def hybrid_search(self, query: str, top_k: int = 5) -> Dict:
        chunk_results = self.search_chunks(query, top_k)
        entity_results = self.search_entities(query, top_k=top_k)
        fact_results = self.search_facts(query, top_k=top_k)
        
        return {
            'chunks': chunk_results,
            'entities': entity_results,
            'facts': fact_results,
            'search_strategy': 'hybrid_empirical'
        }

    def get_stats(self) -> Dict:
        return {
            'chunks_count': self.chunks_collection.count(),
            'entities_count': self.entities_collection.count(),
            'facts_count': self.facts_collection.count(),
            'parent_chunks_cached': len(self.parent_chunks),
            'entity_index_keys': len(self.entity_index),
            'fact_index_keys': len(self.fact_index)
        }

    def clear_all(self):
        self.client.delete_collection("chunks")
        self.client.delete_collection("entities")
        self.client.delete_collection("facts")
        
        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            embedding_function=self.embedding_fn
        )
        self.entities_collection = self.client.get_or_create_collection(
            name="entities",
            embedding_function=self.embedding_fn
        )
        self.facts_collection = self.client.get_or_create_collection(
            name="facts",
            embedding_function=self.embedding_fn
        )
        
        self.parent_chunks.clear()
        self.entity_index.clear()
        self.fact_index.clear()
