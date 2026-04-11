
import requests
import json

def call_qwen_model(messages):
    url = "https://rsm-8430-finalproject.bjlkeng.io/v1/chat/completions"
    
    # YOUR_STUDENT_ID should be your Rotman ID (e.g., "1012136680")
    STUDENT_ID = "1012136680" 
    
    headers = {
        "Authorization": f"Bearer {STUDENT_ID}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen3-30b-a3b-fp8",
        "messages": messages,
        "temperature": 0.1
    }
    
    try:
        # Added headers=headers to the request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status() 
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

def process_query(user_query, retrieved_context, objective, history):
    # This prompt now explicitly handles the 'State' of the conversation
    if retrieved_context == "NONE_FOUND":
        return "I'm sorry, I could not find any official records for that location in the Toronto Open Data database. Please check the spelling or provide a valid Toronto address."
    system_prompt = f"""
    You are the Toronto City Bylaw Agent. 
    Current Category: {objective}
    You are the official City of Toronto Municipal Assistant. 
    1. STRICTOR GROUNDING: Use ONLY the provided context to answer. 
    2. GEOGRAPHIC LIMIT: You only provide information for the City of Toronto. 
    3. If the user asks about Vancouver, Montreal, or any non-Toronto location, politely refuse and state you only serve Toronto.
    4. If the provided context does not mention the specific street or item the user asked for, state clearly that no record was found.
    5. Use ONLY the provided context: {retrieved_context}.
    
    
    
    STRICT OPERATING INSTRUCTIONS:
    1. You have access to a local database of 30,000 city records provided in the 'CONTEXT' below.
    2. If the 'CONTEXT' contains information about a permit, address, or waste item, you MUST use that data to answer the user's question. 
    3. DO NOT say you don't have access to real-time data if the information is present in the context.
    
    
    MULTI-TURN ACTION PROTOCOL:
    - If the user is reporting a Hazard:
        1. Check if the 'USER QUERY' or 'HISTORY' contains a location (Street name/Postal Code).
        2. If NO location is found, you MUST say: "I can help you report this. Could you please provide the street name or nearest intersection?"
        3. If a location is found, acknowledge it and provide a MOCK reference number (e.g., 'SR-2026-X').

    STRICT GUARDRAILS:
    1. Only discuss Toronto municipal services (Hazards, Permits, Waste). 
    2. For out-of-scope queries (legal, politics), politely decline.
    3. Only give answers to questions asked for the locations in chroma_db database.
    
    Citations: Always mention if info came from the 'Hazard', 'Permit', or 'Waste' database.
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Context Memory: Injecting previous turns so the LLM remembers the location it asked for
    for h in history[-4:]: 
        messages.append(h)
    
    messages.append({
        "role": "user", 
        "content": f"CONTEXT: {retrieved_context}\n\nUSER QUERY: {user_query}"
    })

    return call_qwen_model(messages)
