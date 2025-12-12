import streamlit as st
from src.empirical_rag_pipeline import EmpiricalRAGPipeline
import os

st.set_page_config(page_title="RAG System with Empirical Data Modelling", layout="wide")

DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "documents")

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = EmpiricalRAGPipeline(DOCUMENTS_PATH)
    st.session_state.initialized = False
    st.session_state.chat_history = []

st.title("Document Q&A with Empirical Data Modelling")

with st.sidebar:
    st.header("Document Management")
    
    if st.button("Load/Reload Documents"):
        with st.spinner("Loading documents with Empirical Data Modelling..."):
            result = st.session_state.rag_pipeline.reload_documents()
            st.session_state.initialized = result['success']
            
            if result['success']:
                st.success(f"Loaded {result['document_count']} documents")
                st.info(f"Total Chunks: {result['total_chunks']}")
                st.info(f"Entities Extracted: {result['total_entities']}")
                st.info(f"Facts Extracted: {result['total_facts']}")
                
                if result.get('entities_by_type'):
                    st.subheader("Entities by Type")
                    for etype, count in result['entities_by_type'].items():
                        st.write(f"- {etype}: {count}")
            else:
                st.error(result['message'])
    
    st.divider()
    
    if st.session_state.initialized:
        stats = st.session_state.rag_pipeline.get_db_stats()
        st.subheader("Vector DB Stats")
        vs_stats = stats.get('vector_store_stats', {})
        st.caption(f"Chunks: {vs_stats.get('chunks_count', 0)}")
        st.caption(f"Entities: {vs_stats.get('entities_count', 0)}")
        st.caption(f"Facts: {vs_stats.get('facts_count', 0)}")
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.header("Supported Formats")
    st.write("PDF, DOCX, Excel, CSV, TXT")

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['question'])
    with st.chat_message("assistant"):
        st.write(chat['answer'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if "SLM" in chat['model_used']:
                st.success(f"Model: {chat['model_used']}")
            else:
                st.info(f"Model: {chat['model_used']}")
        with col2:
            st.caption(f"Complexity: {chat['complexity_score']:.2f}")
        with col3:
            st.caption(f"Relevance: {chat['relevance_score']:.2f}")
        
        if chat.get('empirical_analysis'):
            with st.expander("Empirical Data Model Analysis"):
                analysis = chat['empirical_analysis']
                
                st.markdown("**Search Strategy:** " + analysis.get('search_strategy', 'N/A'))
                st.markdown("**Collections Searched:** " + ", ".join(analysis.get('collections_searched', [])))
                
                summary = analysis.get('summary', {})
                st.markdown(f"**Entities Matched:** {summary.get('entities_matched', 0)}")
                st.markdown(f"**Facts Matched:** {summary.get('facts_matched', 0)}")
                st.markdown(f"**Entity Types:** {', '.join(summary.get('entity_types', []))}")
                
                if analysis.get('entities_found'):
                    st.markdown("---")
                    st.markdown("**Matched Entities:**")
                    for ent in analysis['entities_found'][:5]:
                        st.write(f"- [{ent['type']}] {ent['value']} (relevance: {ent['relevance']})")
                
                if analysis.get('facts_found'):
                    st.markdown("---")
                    st.markdown("**Matched Facts:**")
                    for fact in analysis['facts_found'][:5]:
                        st.write(f"- {fact['relationship']}")
                        st.caption(f"  Evidence: {fact['evidence']}")
                
                if analysis.get('relationship_graph'):
                    st.markdown("---")
                    st.markdown("**Relationship Graph:**")
                    for rel in analysis['relationship_graph'][:5]:
                        st.code(f"{rel['from']} --[{rel['relation']}]--> {rel['to']}")

question = st.chat_input("Ask a question...")

if question:
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing with Empirical Data Model..."):
            result = st.session_state.rag_pipeline.query(question)
            st.write(result['answer'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if "SLM" in result['model_used']:
                    st.success(f"Model: {result['model_used']}")
                else:
                    st.info(f"Model: {result['model_used']}")
            with col2:
                st.caption(f"Complexity: {result['complexity_score']:.2f}")
            with col3:
                st.caption(f"Relevance: {result['relevance_score']:.2f}")
            
            if result.get('sources'):
                with st.expander("Sources"):
                    for source in result['sources']:
                        st.write(f"- {source['filename']} (Relevance: {source['relevance']:.2f})")
            
            if result.get('empirical_analysis'):
                with st.expander("Empirical Data Model Analysis"):
                    analysis = result['empirical_analysis']
                    
                    st.markdown("**Search Strategy:** " + analysis.get('search_strategy', 'N/A'))
                    st.markdown("**Collections Searched:** " + ", ".join(analysis.get('collections_searched', [])))
                    
                    summary = analysis.get('summary', {})
                    st.markdown(f"**Entities Matched:** {summary.get('entities_matched', 0)}")
                    st.markdown(f"**Facts Matched:** {summary.get('facts_matched', 0)}")
                    st.markdown(f"**Entity Types:** {', '.join(summary.get('entity_types', []))}")
                    
                    if analysis.get('entities_found'):
                        st.markdown("---")
                        st.markdown("**Matched Entities:**")
                        for ent in analysis['entities_found'][:5]:
                            st.write(f"- [{ent['type']}] {ent['value']} (relevance: {ent['relevance']})")
                    
                    if analysis.get('facts_found'):
                        st.markdown("---")
                        st.markdown("**Matched Facts:**")
                        for fact in analysis['facts_found'][:5]:
                            st.write(f"- {fact['relationship']}")
                            st.caption(f"  Evidence: {fact['evidence']}")
                    
                    if analysis.get('relationship_graph'):
                        st.markdown("---")
                        st.markdown("**Relationship Graph:**")
                        for rel in analysis['relationship_graph'][:5]:
                            st.code(f"{rel['from']} --[{rel['relation']}]--> {rel['to']}")
    
    st.session_state.chat_history.append({
        'question': question,
        'answer': result['answer'],
        'model_used': result['model_used'],
        'complexity_score': result['complexity_score'],
        'relevance_score': result['relevance_score'],
        'sources': result.get('sources', []),
        'empirical_analysis': result.get('empirical_analysis', {})
    })
    st.rerun()
