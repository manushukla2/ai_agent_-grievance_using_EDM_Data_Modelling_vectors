from typing import Dict
from src.document_loader import DocumentLoader
from src.chunking import HierarchicalChunker
from src.vector_store import VectorStore
from src.complexity import ComplexityAnalyzer
from src.llm_handler import LLMHandler


class RAGPipeline:
    def __init__(self, documents_path: str):
        self.documents_path = documents_path
        self.document_loader = DocumentLoader(documents_path)
        self.chunker = HierarchicalChunker()
        self.vector_store = VectorStore()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.llm_handler = LLMHandler()
        self.is_initialized = False
    
    def initialize(self) -> Dict:
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            return {
                'success': False,
                'message': 'No documents found in the documents folder',
                'document_count': 0
            }
        
        hierarchical_data = self.chunker.create_hierarchical_chunks(documents)
        self.vector_store.add_documents(hierarchical_data)
        self.is_initialized = True
        
        return {
            'success': True,
            'message': 'Documents loaded and indexed successfully',
            'document_count': len(documents),
            'parent_chunks': len(hierarchical_data['parent_chunks']),
            'child_chunks': len(hierarchical_data['child_chunks'])
        }
    
    def query(self, question: str) -> Dict:
        if not self.is_initialized:
            init_result = self.initialize()
            if not init_result['success']:
                return {
                    'answer': 'No documents available. Please add documents to the documents folder.',
                    'model_used': 'None',
                    'complexity_score': 0,
                    'relevance_score': 0,
                    'sources': []
                }
        
        model_type, complexity_score, complexity_reason = self.complexity_analyzer.analyze(question)
        search_results = self.vector_store.search(question, top_k=5)
        
        if not search_results:
            return {
                'answer': 'No relevant information found in the documents.',
                'model_used': model_type,
                'complexity_score': complexity_score,
                'complexity_reason': complexity_reason,
                'relevance_score': 0,
                'sources': []
            }
        
        filtered_results = [r for r in search_results if r['relevance_score'] > 0.3]
        if not filtered_results:
            filtered_results = search_results[:2]
        
        context = "\n\n---\n\n".join([result['parent_text'] for result in filtered_results])
        avg_relevance = sum([result['relevance_score'] for result in filtered_results]) / len(filtered_results)
        
        use_llm = model_type == "LLM"
        response = self.llm_handler.generate(question, context, use_llm)
        
        sources = [{'filename': r['filename'], 'relevance': round(r['relevance_score'], 3)} for r in filtered_results]
        
        return {
            'answer': response['answer'],
            'model_used': response['model_used'],
            'model_name': response['model_name'],
            'complexity_score': round(complexity_score, 3),
            'complexity_reason': complexity_reason,
            'relevance_score': round(avg_relevance, 3),
            'sources': sources,
            'success': response['success']
        }
    
    def reload_documents(self) -> Dict:
        self.vector_store.clear()
        self.is_initialized = False
        return self.initialize()
