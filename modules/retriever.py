
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

# Use a relative path so it finds 'chroma_db' inside your project folder
def load_brain(db_dir="./chroma_db"):
    """Loads the 30k record database from the local project directory."""
    
    # Check if path exists to avoid confusing errors
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"ChromaDB not found at {os.path.abspath(db_dir)}. Check your folder structure!")

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
    vectorstore = Chroma(
        persist_directory=db_dir, 
        embedding_function=embeddings,
        collection_name="toronto_city_ai" 
    )
    return vectorstore

def get_city_context(query, vectorstore):
    # Search with scores
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    if not results:
        return "NONE_FOUND", "general"

    best_doc, score = results[0]

    # STRICT THRESHOLD: If the distance is too high, it's a hallucination
    # Adjust 0.8 based on your specific embedding model's sensitivity
    if score > 0.9: 
        return "NONE_FOUND", "general"

    return best_doc.page_content, best_doc.metadata.get("objective", "general")
    
    # 2. Grab the best result
    best_doc, score = results[0]
    
    # 3. RELAX THE THRESHOLD: 
    # In ChromaDB, lower scores are better. 
    # If the score is > 1.2, it's usually a bad match.
    if score > 1.5: 
        return "No confident matches found in the database.", "general"

    # 4. Extract the data and the objective
    context_text = best_doc.page_content
    objective = best_doc.metadata.get("objective", "general")
    
    return context_text, objective

