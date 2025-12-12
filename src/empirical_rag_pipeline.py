from typing import Dict, List
from src.document_loader import DocumentLoader
from src.chunking import HierarchicalChunker
from src.empirical_vector_store import EmpiricalVectorStore
from src.entity_extractor import EntityExtractor
from src.complexity import ComplexityAnalyzer
from src.llm_handler import LLMHandler


class EmpiricalRAGPipeline:
    def __init__(self, documents_path: str):
        self.documents_path = documents_path
        self.document_loader = DocumentLoader(documents_path)
        self.chunker = HierarchicalChunker()
        self.vector_store = EmpiricalVectorStore()
        self.entity_extractor = EntityExtractor()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.llm_handler = LLMHandler()
        self.is_initialized = False
        self.indexing_stats = {}
    
    def initialize(self) -> Dict:
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            return {
                'success': False,
                'message': 'No documents found in the documents folder',
                'document_count': 0
            }
        
        self.vector_store.clear_all()
        
        total_entities = 0
        total_facts = 0
        total_chunks = 0
        entities_by_type = {}
        
        for doc in documents:
            filename = doc['filename']
            content = doc['content']
            
            chunks = self.chunker.create_chunks_for_document(content, filename)
            total_chunks += len(chunks)
            
            for chunk in chunks:
                chunk_id = f"{filename}_{chunk['chunk_id']}"
                
                self.vector_store.add_chunks([chunk], filename)
                
                entities = self.entity_extractor.extract_entities(chunk['parent_text'])
                total_entities += len(entities)
                
                for entity in entities:
                    entities_by_type[entity.entity_type] = entities_by_type.get(entity.entity_type, 0) + 1
                
                self.vector_store.add_entities(entities, chunk_id, filename, chunk['child_text'])
                
                facts = self.entity_extractor.extract_facts(chunk['parent_text'], entities)
                total_facts += len(facts)
                
                self.vector_store.add_facts(facts, chunk_id, filename)
        
        self.is_initialized = True
        
        self.indexing_stats = {
            'document_count': len(documents),
            'total_chunks': total_chunks,
            'total_entities': total_entities,
            'total_facts': total_facts,
            'entities_by_type': entities_by_type
        }
        
        return {
            'success': True,
            'message': 'Documents loaded with Empirical Data Modelling',
            'document_count': len(documents),
            'total_chunks': total_chunks,
            'total_entities': total_entities,
            'total_facts': total_facts,
            'entities_by_type': entities_by_type
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
                    'sources': [],
                    'empirical_analysis': {}
                }
        
        model_type, complexity_score, complexity_reason = self.complexity_analyzer.analyze(question)
        
        hybrid_results = self.vector_store.hybrid_search(question, top_k=5)
        
        chunk_results = hybrid_results['chunks']
        entity_results = hybrid_results['entities']
        fact_results = hybrid_results['facts']
        
        empirical_analysis = self._build_empirical_analysis(
            question, entity_results, fact_results
        )
        
        if not chunk_results:
            return {
                'answer': 'No relevant information found in the documents.',
                'model_used': model_type,
                'complexity_score': complexity_score,
                'complexity_reason': complexity_reason,
                'relevance_score': 0,
                'sources': [],
                'empirical_analysis': empirical_analysis
            }
        
        filtered_chunks = [r for r in chunk_results if r['relevance_score'] > 0.3]
        if not filtered_chunks:
            filtered_chunks = chunk_results[:2]
        
        context_parts = []
        
        for chunk in filtered_chunks:
            context_parts.append(chunk['parent_text'])
        
        if fact_results:
            fact_context = "\n\nRelevant Facts:\n"
            for fact in fact_results[:5]:
                if fact['relevance_score'] > 0.3:
                    fact_context += f"- {fact['subject']} {fact['predicate']} {fact['object']}\n"
            if len(fact_context) > 20:
                context_parts.append(fact_context)
        
        context = "\n\n---\n\n".join(context_parts)
        
        avg_relevance = sum([r['relevance_score'] for r in filtered_chunks]) / len(filtered_chunks)
        
        use_llm = model_type == "LLM"
        response = self.llm_handler.generate(question, context, use_llm)
        
        sources = [{'filename': r['filename'], 'relevance': round(r['relevance_score'], 3)} for r in filtered_chunks]
        
        return {
            'answer': response['answer'],
            'model_used': response['model_used'],
            'model_name': response['model_name'],
            'complexity_score': round(complexity_score, 3),
            'complexity_reason': complexity_reason,
            'relevance_score': round(avg_relevance, 3),
            'sources': sources,
            'success': response['success'],
            'empirical_analysis': empirical_analysis
        }
    
    def _build_empirical_analysis(self, question: str, entity_results: List[Dict], fact_results: List[Dict]) -> Dict:
        analysis = {
            'search_strategy': 'Hybrid Empirical Data Model',
            'collections_searched': ['chunks', 'entities', 'facts'],
            'entities_found': [],
            'facts_found': [],
            'entity_types_matched': set(),
            'relationship_graph': []
        }
        
        for entity in entity_results[:10]:
            if entity['relevance_score'] > 0.25:
                analysis['entities_found'].append({
                    'type': entity['entity_type'],
                    'value': entity['entity_text'],
                    'relevance': round(entity['relevance_score'], 3)
                })
                analysis['entity_types_matched'].add(entity['entity_type'])
        
        for fact in fact_results[:10]:
            if fact['relevance_score'] > 0.25:
                analysis['facts_found'].append({
                    'relationship': f"{fact['subject']} --[{fact['predicate']}]--> {fact['object']}",
                    'evidence': fact['source_text'][:100] + "..." if len(fact['source_text']) > 100 else fact['source_text'],
                    'relevance': round(fact['relevance_score'], 3)
                })
                
                analysis['relationship_graph'].append({
                    'from': fact['subject'],
                    'to': fact['object'],
                    'relation': fact['predicate']
                })
        
        analysis['entity_types_matched'] = list(analysis['entity_types_matched'])
        
        analysis['summary'] = {
            'entities_matched': len(analysis['entities_found']),
            'facts_matched': len(analysis['facts_found']),
            'entity_types': analysis['entity_types_matched'],
            'has_relationships': len(analysis['relationship_graph']) > 0
        }
        
        return analysis
    
    def get_db_stats(self) -> Dict:
        return {
            'indexing_stats': self.indexing_stats,
            'vector_store_stats': self.vector_store.get_stats()
        }
    
    def reload_documents(self) -> Dict:
        self.vector_store.clear_all()
        self.is_initialized = False
        return self.initialize()
