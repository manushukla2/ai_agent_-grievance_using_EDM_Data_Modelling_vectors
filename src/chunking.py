from typing import List, Dict
import uuid


class HierarchicalChunker:
    def __init__(self, parent_chunk_size: int = 2000, child_chunk_size: int = 500, overlap: int = 100):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap
    
    def create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def create_hierarchical_chunks(self, documents: List[Dict]) -> Dict:
        parent_chunks = {}
        child_chunks = []
        child_to_parent_map = {}
        
        for doc in documents:
            content = doc['content']
            filename = doc['filename']
            
            parents = self.create_chunks(content, self.parent_chunk_size, self.overlap)
            
            for parent_text in parents:
                parent_id = str(uuid.uuid4())
                parent_chunks[parent_id] = {
                    'text': parent_text,
                    'filename': filename
                }
                
                children = self.create_chunks(parent_text, self.child_chunk_size, self.overlap // 2)
                
                for child_text in children:
                    child_id = str(uuid.uuid4())
                    child_chunks.append({
                        'id': child_id,
                        'text': child_text,
                        'parent_id': parent_id,
                        'filename': filename
                    })
                    child_to_parent_map[child_id] = parent_id
        
        return {
            'parent_chunks': parent_chunks,
            'child_chunks': child_chunks,
            'child_to_parent_map': child_to_parent_map
        }

    def create_chunks_for_document(self, content: str, filename: str) -> List[Dict]:
        chunks = []
        parents = self.create_chunks(content, self.parent_chunk_size, self.overlap)
        
        for i, parent_text in enumerate(parents):
            parent_id = f"{filename}_parent_{i}"
            
            children = self.create_chunks(parent_text, self.child_chunk_size, self.overlap // 2)
            
            for j, child_text in enumerate(children):
                chunk_id = f"parent_{i}_child_{j}"
                chunks.append({
                    'chunk_id': chunk_id,
                    'parent_id': parent_id,
                    'child_text': child_text,
                    'parent_text': parent_text
                })
        
        return chunks
