
import os
import json
import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def build_30k_rag(project_path="."):
    """
    Builds a 30,000 record vector database from Toronto Open Data.
    Saves to the local project directory.
    """
    data_dir = os.path.join(project_path, "data")
    final_db_dir = os.path.join(project_path, "chroma_db")
    
    # UPDATED: Use a local temp folder instead of /content/
    local_db_dir = os.path.join(project_path, "temp_indexing_folder") 

    documents = []
    
    # --- 1. 311 HAZARDS ---
    csv_311 = os.path.join(data_dir, "311_service_requests.csv")
    if os.path.exists(csv_311):
        print("🚧 Ingesting 25,000 311 Records...")
        # Added low_memory=False for better local performance
        df = pd.read_csv(csv_311, encoding="latin1", on_bad_lines='skip', low_memory=False).head(25000)
        for _, row in df.iterrows():
            issue = str(row.get('Service Request Type') or 'Hazard')
            documents.append(Document(
                page_content=f"OBJECTIVE: Hazard Reporter | ISSUE: {issue}. Details: {row.get('Description', 'Standard Protocol')}",
                metadata={"objective": "hazard_reporter"}
            ))

    # --- 2. PERMITS ---
    json_permits = os.path.join(data_dir, "Cleared Building Permits since 2017.json")
    if os.path.exists(json_permits):
        print("🏗️ Ingesting 5,000 Permit Records...")
        with open(json_permits, 'r', encoding="utf-8") as f:
            permit_data = json.load(f)
        for r in permit_data[:5000]:
            num, name = str(r.get('STREET_NUM') or ''), str(r.get('STREET_NAME') or '')
            addr = f"{num} {name}".strip().upper()
            raw_desc = str(r.get('DESCRIPTION') or '')
            documents.append(Document(
                page_content=f"OBJECTIVE: Permit Screener | LOCATION: {addr}. Work: {r.get('WORK')}. Info: {raw_desc[:100]}",
                metadata={"objective": "permit_screener", "street": name}
            ))

    # --- 3. WASTE WIZARD ---
    json_waste = os.path.join(data_dir, "Waste Wizard Lookup Table.json")
    if os.path.exists(json_waste):
        print("♻️ Ingesting Waste Wizard...")
        with open(json_waste, 'r', encoding="utf-8") as f:
            waste_data = json.load(f)
        for r in waste_data:
            item = str(r.get('item') or 'item')
            instr = " ".join(r.get('instructions', [])) if isinstance(r.get('instructions'), list) else ""
            documents.append(Document(
                page_content=f"OBJECTIVE: Collection Lookup | ITEM: {item}. Bin: {r.get('category')}. Instructions: {instr}",
                metadata={"objective": "collection_lookup", "item": item}
            ))

    
    # Update this part:
    
    STUDENT_ID = "1012136680" 

    print("🧠 Connecting to Professor's A2 Embedding Endpoint...")
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    openai_api_key=STUDENT_ID, 
    openai_api_base="https://rsm-8430-a2.bjlkeng.io/v1",
    # CRITICAL FIX: This prevents the library from pre-tokenizing the text
    # and forces it to send raw strings to the professor's server.
    check_embedding_ctx_length=False,
    # Just in case, we also set chunk_size to 1 to ensure strings are sent cleanly
    chunk_size=100
    )
    
    if os.path.exists(local_db_dir): shutil.rmtree(local_db_dir)
    
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=local_db_dir,
        collection_name="toronto_city_ai"
    )

    print("💾 Finalizing local ChromaDB...")
    if os.path.exists(final_db_dir): shutil.rmtree(final_db_dir)
    shutil.copytree(local_db_dir, final_db_dir)
    
    # Cleanup temp folder
    shutil.rmtree(local_db_dir)
    
    print("🎉 MASTER 30K RAG REBUILT SUCCESSFULLY!")
    return vectorstore

if __name__ == "__main__":
    build_30k_rag(project_path=".")
