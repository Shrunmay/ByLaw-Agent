# Toronto City AI ByLaw Agent
### Rotman School of Management | RSM8431 Project
#### By Shrunmay Shinde, Austin Li, Sherry Liu, Ivy Pan, Rebecca Lee

This repository contains a full-stack, conversational AI agent designed to assist citizens with non-emergency City of Toronto municipal services. The agent is strictly grounded in over **30,000 official records** from Toronto Open Data, covering building permits, waste disposal protocols, and non-emergency service requests (e.g., potholes, graffiti).

---

## 🚀 Project Overview

The Toronto City AI Agent addresses the growing challenge of citizen information overload by providing a streamlined, natural language interface to disparate municipal databases. 

Built using a modular Python architecture, the agent utilizes:
* **Vector Embeddings & Semantic Search** (via ChromaDB) to quickly surface the correct by-law or municipal record from a 30,000+ record knowledge base.
* **Large Language Models** (via the Qwen API) to process complex, multi-turn conversations and enforce strict municipal personas.
* **Streamlit** to deliver a production-quality, responsive user interface.

### Key Capabilities
* **Intent Routing:** Automatically classifies queries into Building Permits, Service Requests, or Waste Management, routing them to the correct retrieval logic.
* **Hallucination Guardrails:** Utilizes similarity thresholding to refuse "Nearest Neighbor" results for non-existent locations (e.g., "Pikachu Street"), preventing hallucinations while maintaining strict **Toronto-only grounding**.
* **Source Attribution:** Every fact-based answer includes the direct source from the Toronto Open Data portal.

---

## 🏗️ Technical Architecture & Project Structure

The project follows a modular pattern, where the main app (`app.py`) orchestrates the interactions between specialized modules.

```bash
BYLAW_AGENT/
├── chroma_db/                  # The production vector database (DO NOT DELETE)
│   ├── [UUID folder]/         # Vector segment data
│   └── chroma.sqlite3          # DB metadata and structure
├── data/                       # Source Toronto Open Data files (CSV/JSON)
│   ├── 311_service_requests.csv # Non-emergency service records
│   ├── Cleared_Building_Permits.json # Building permit database
│   └── Waste_Wizard_Lookup_Table.json # Waste sorting protocol
├── modules/                    # Specialized application logic (Package)
│   ├── __init__.py
│   ├── ingestor.py             # Parses data and builds the Vector DB
│   ├── retriever.py            # Handles semantic search and thresholding
│   └── llm_interface.py        # Manages system prompting, memory, and LLM calls
├── notebooks/                  # Experimental and Evaluation logic
│   ├── EDA_and_Indexing.ipynb  # Initial data analysis and indexing test
│   └── evaluation_metrics.py    # Industry-standard RAG metric calculator
├── .gitignore                  # Files to exclude from version control (e.g., venv/)
├── app.py                      # Main Streamlit user interface
├── evaluation_report.csv       # Output of the final evaluation run
├── README.md                   # This project documentation
├── requirements.txt            # Project dependencies
└── Test_Cases.txt               # A sample of the 15 test cases

```
### 💾 Data Management Note
Due to GitHub's file size limitations (>50MB), the raw `chroma_db` vector store and large JSON source files are not included in this repository.

Due to the large size of our knowledge base (30,000+ records), the local vector store exceeds GitHub's file size limits. We provide two ways to access the project data:

### 🟢 Option 1: Quick Start (Pre-indexed Database)
To run the agent immediately without waiting for the embedding process:
1. **Download** the vector store: [Download chroma_db.zip](https://drive.google.com/drive/folders/1BEsA8wt1cOx6wyyXF3-dvVEL90RpvVWl?usp=sharing)
2. **Save** the `chroma_db` folder into the project root directory (`/ByLaw_Agent/`).
3. Ensure your directory structure looks like this:
   ```text
   ByLaw_Agent/
   ├── chroma_db/  <-- (Extracted folder here)
   ├── app.py
   └── ...

### 🟢 Option 2: To Reconstruct the Knowledge Base:
1. Download the following datasets from **Toronto Open Data**:
   - [Building Permits](https://open.toronto.ca/dataset/building-permits-cleared-permits/)
   - [311 Service Requests](https://open.toronto.ca/dataset/311-service-requests-customer-initiated/)
   - [Waste Wizard](https://open.toronto.ca/dataset/waste-wizard-lookup-table/)
2. Place the files in the `/data` directory.
3. Run the ingestion script to rebuild the vector database:
   ```bash
   python -c "from modules.ingestor import build_30k_rag; build_30k_rag(project_path='.')"


   ## 📊 Evaluation & Performance Metrics

The agent's reliability was audited using a test set of **15 representative "Golden" test cases**, covering single-turn and multi-turn interactions, intent routing, and safety guardrails.

### 📈 RAG Triad Results
The following metrics were calculated using the Qwen-30B model as a reasoning judge to ensure the agent's responses align with the **Toronto Open Data** knowledge base:

| Metric | Score | Key Takeaway |
| :--- | :--- | :--- |
| **Faithfulness** | **0.70 / 1.0** | Reflects the agent's use of conversational "connective tissue" and general municipal safety procedures beyond raw data points. |
| **Answer Relevance** | **0.71 / 1.0** | High score indicates that user intent is accurately identified and addressed with helpful, actionable advice. |
| **Context Precision** | **0.44 / 1.0** | Demonstrates the retriever's ability to successfully surface relevant documents from a 30k record pool. |

### 📂 Detailed Case Analysis
The full granular breakdown of every test case—including the specific queries, retrieved context, and individual scores—is stored in:
👉 **`evaluation_report.csv`**

This file includes our "Failure Case Analysis," where we identify how the agent handles out-of-scope queries (e.g., non-existent streets) and how it gracefully degrades when the specific record is not present in the 5k-record subset.