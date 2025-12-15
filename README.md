# RAG System with Entity Data Modelling

A document Q&A system that goes beyond simple text matching. It extracts **entities** (people, dates, IDs) and **facts** (relationships between entities) from your documents, then uses this structured knowledge to give more accurate answers.

---

## What is Entity Data Modelling (EDM)?

Traditional RAG systems just split documents into chunks and search by text similarity. This often misses precise information.

**EDM adds a layer of understanding:**

```
Traditional RAG:
Document → Chunks → Vector Search → Answer

EDM-Enhanced RAG:
Document → Chunks → Entity Extraction → Fact Extraction → 3 Vector Collections → Hybrid Search → Answer
                         ↓                    ↓
                    "Mr. Sharma"        "Mr. Sharma works in HR"
                    "15-Jan-2024"       "Case #123 filed on 15-Jan-2024"
                    "Case #123"
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (app.py)                       │
│                    Chat Interface + EDM Analysis View               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 EMPIRICAL RAG PIPELINE                              │
│                 (src/empirical_rag_pipeline.py)                     │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Document   │  │  Complexity  │  │     LLM      │              │
│  │    Loader    │  │   Analyzer   │  │   Handler    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    EMPIRICAL DATA MODELLING LAYER                    │
│                                                                      │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐     │
│  │  Hierarchical  │    │    Entity      │    │     Fact       │     │
│  │   Chunker      │    │   Extractor    │    │   Extractor    │     │
│  │                │    │                │    │                │     │
│  │ Parent: 2000   │    │ PERSON, DATE   │    │ Subject-       │     │
│  │ Child: 500     │    │ ID, STATUS     │    │ Predicate-     │     │
│  │ chars          │    │ DEPARTMENT     │    │ Object         │     │
│  └────────────────┘    └────────────────┘    └────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     CHROMADB VECTOR STORE                            │
│                  (3 Separate Collections)                            │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │   CHUNKS     │   │  ENTITIES    │   │    FACTS     │             │
│  │  Collection  │   │  Collection  │   │  Collection  │             │
│  │              │   │              │   │              │             │
│  │ Child text   │   │ Entity +     │   │ Subject +    │             │
│  │ + Parent     │   │ Context      │   │ Predicate +  │             │
│  │ mapping      │   │              │   │ Object       │             │
│  └──────────────┘   └──────────────┘   └──────────────┘             │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          HYBRID SEARCH
                    (Searches all 3 collections)
```

---

## How EDM Works (Step by Step)

### Step 1: Document Loading
Files from `documents/` folder are loaded. Supports PDF, DOCX, Excel, CSV, TXT.

### Step 2: Hierarchical Chunking
Each document is split into:
- **Parent Chunks** (2000 chars): Provide complete context
- **Child Chunks** (500 chars): Used for precise search matching

```
Original Document
       │
       ▼
┌──────────────────────────────────┐
│      Parent Chunk (2000 chars)   │
│  ┌─────────┐ ┌─────────┐ ┌─────┐ │
│  │ Child 1 │ │ Child 2 │ │ ... │ │
│  │ 500char │ │ 500char │ │     │ │
│  └─────────┘ └─────────┘ └─────┘ │
└──────────────────────────────────┘
```

**Why?** Child chunks give precise matches. Parent chunks give complete context to LLM.

### Step 3: Entity Extraction
The system scans each chunk and extracts structured entities:

| Entity Type | Examples | Pattern Used |
|-------------|----------|--------------|
| PERSON | Mr. Sharma, Dr. Gupta | Title + Name regex |
| DATE | 15-Jan-2024, 2024/01/15 | Multiple date formats |
| ID | GR-001, CASE-123, EMP-456 | ID patterns |
| STATUS | pending, approved, rejected | Keyword matching |
| DEPARTMENT | HR, Finance, IT | Department keywords |
| MONEY | Rs. 50,000, $1000 | Currency patterns |

### Step 4: Fact Extraction
The system finds relationships between entities in the same sentence:

```
Sentence: "Mr. Sharma from HR filed case GR-001 on 15-Jan-2024, status pending"

Extracted Facts:
  - Mr. Sharma --[belongs_to]--> HR
  - GR-001 --[has_status]--> pending
  - GR-001 --[dated]--> 15-Jan-2024
```

### Step 5: Storage in 3 Collections
Everything goes into ChromaDB with embeddings:

1. **Chunks Collection**: Searchable child chunks with parent mapping
2. **Entities Collection**: Each entity with its surrounding context
3. **Facts Collection**: Each relationship as searchable text

### Step 6: Hybrid Search
When you ask a question:
1. Search **Chunks** for relevant text passages
2. Search **Entities** for matching entities
3. Search **Facts** for matching relationships
4. Combine results and build context for LLM

---

## Smart Model Routing

The system automatically chooses the right model:

| Question Type | Model Used | Example |
|---------------|------------|---------|
| Simple factual | SLM (IBM Granite 400M) | "What is grievance?" |
| Complex analysis | LLM (Phi-3 3.8B) | "Compare all pending cases and their reasons" |

**How it decides:**
- Counts question complexity (word count, parts, keywords)
- Simple keywords → SLM (fast, ~150 tokens)
- Complex keywords (compare, analyze, explain) → LLM (detailed, ~200 tokens)

---

## Project Structure

```
greviance_poc/
├── app.py                      # Streamlit UI
├── documents/                  # Put your documents here
├── models/                     # Downloaded models (SLM/LLM)
├── chroma_empirical_db/        # Vector database storage
├── requirements.txt            # Python dependencies
├── scripts/
│   └── download_models.py      # Pre-download models
└── src/
    ├── document_loader.py      # Loads PDF, DOCX, Excel, CSV, TXT
    ├── chunking.py             # Hierarchical parent-child chunking
    ├── entity_extractor.py     # Extracts entities and facts
    ├── empirical_vector_store.py  # 3-collection ChromaDB
    ├── empirical_rag_pipeline.py  # Main orchestration
    ├── complexity.py           # SLM/LLM routing logic
    └── llm_handler.py          # Model loading and inference
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models (one-time, ~5GB)
python scripts/download_models.py

# 3. Add documents to documents/ folder

# 4. Run the app
python -m streamlit run app.py
```

---

## What You See in the UI

### Sidebar
- **Load/Reload Documents**: Processes documents with EDM
- **Vector DB Stats**: Shows chunks, entities, facts count
- **Entities by Type**: Breakdown of extracted entities

### Chat Response
- **Answer**: Generated response
- **Model Used**: SLM or LLM (highlighted)
- **Complexity Score**: Why that model was chosen
- **Relevance Score**: How well sources matched

### Empirical Data Model Analysis (Expandable)
- **Search Strategy**: Hybrid Empirical Data Model
- **Collections Searched**: chunks, entities, facts
- **Matched Entities**: Which entities matched your query
- **Matched Facts**: Which relationships were found
- **Relationship Graph**: Visual representation of facts

---

## Example Query Flow

**Question:** "What is the status of Mr. Sharma's grievance?"

```
1. COMPLEXITY ANALYSIS
   └── Simple question → Use SLM

2. HYBRID SEARCH
   ├── Chunks: Found 3 relevant passages
   ├── Entities: Found "Mr. Sharma" (PERSON), matched with 0.85 relevance
   └── Facts: Found "Mr. Sharma --[has_status]--> pending"

3. CONTEXT BUILDING
   └── Parent chunks + Relevant facts combined

4. LLM GENERATION
   └── SLM generates answer using context

5. RESPONSE
   ├── Answer: "Mr. Sharma's grievance is currently pending..."
   ├── Model: SLM (IBM Granite 400M)
   └── EDM Analysis: Shows matched entities and facts
```

---

## Why EDM Improves Accuracy

| Scenario | Traditional RAG | With EDM |
|----------|-----------------|----------|
| "Status of case GR-001?" | Searches for similar text, may find wrong case | Directly matches ID entity, finds exact status |
| "Who filed grievance in January?" | May miss if wording differs | Searches DATE entities for January dates |
| "All HR department issues" | Keyword match, incomplete | Finds all DEPARTMENT:HR entities |
| "What did Mr. Sharma report?" | May confuse with other Sharmas | Entity-level matching with context |

---

## Models Used

| Model | Type | Size | Use Case |
|-------|------|------|----------|
| IBM Granite 3.0 1B A400M | SLM | ~400M active params | Simple factual questions |
| Microsoft Phi-3 Mini 4K | LLM | 3.8B params | Complex analysis questions |
| all-MiniLM-L6-v2 | Embeddings | 80MB | Vector embeddings for search |

---

## Key Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| Parent chunk size | 2000 chars | chunking.py |
| Child chunk size | 500 chars | chunking.py |
| Chunk overlap | 100 chars | chunking.py |
| Top-K retrieval | 5 | empirical_rag_pipeline.py |
| Relevance threshold | 0.3 | empirical_rag_pipeline.py |
| SLM max tokens | 150 | llm_handler.py |
| LLM max tokens | 200 | llm_handler.py |

---

## Limitations

1. Entity extraction uses regex patterns, not ML-based NER
2. Fact extraction is rule-based (same-sentence co-occurrence)
3. Models run on CPU (slower but works without GPU)
4. Memory usage can be high with large documents
