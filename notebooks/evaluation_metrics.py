import os
import json
import pandas as pd
from modules.retriever import load_brain, get_city_context
from modules.llm_interface import process_query

# Configuration
STUDENT_ID = "1012136680"
EVAL_RESULTS_FILE = "evaluation_report.csv"

# The 15 Golden Test Cases
TEST_CASES = [
    {"id": 1, "type": "Hazard", "turns": ["I'd like to report a fallen tree branch blocking the sidewalk at Dundas Street and Spadina Ave."]},
    {"id": 2, "type": "Hazard", "turns": ["There is an exposed electrical wire near the park entrance at St George Street."]},
    {"id": 3, "type": "Hazard", "turns": ["There is a dangerous sign hanging by a thread over the sidewalk.", "It's at 75 Broadway Ave, hanging off the second-floor balcony."]},
    {"id": 4, "type": "Hazard", "turns": ["The traffic lights are completely out at this intersection.", "I am at Front Street and University Ave."]},
    {"id": 5, "type": "Hazard", "turns": ["I need to report a flooded street.", "It's right at the corner of Yonge and Bloor."]},
    {"id": 6, "type": "Permit", "turns": ["I'm looking for the status of the building permit at 875 St Clarens."]},
    {"id": 7, "type": "Permit", "turns": ["Are there any active permits for Interior Alterations on St George Street?"]},
    {"id": 8, "type": "Permit", "turns": ["What kind of construction projects are happening on Dundas Street right now?"]},
    {"id": 9, "type": "Waste", "turns": ["I have a pizza box that's covered in grease and some leftover crusts. Which bin does it go in?"]},
    {"id": 10, "type": "Waste", "turns": ["How do I dispose of old cans of half-full oil-based paint and some paint thinner?"]},
    {"id": 11, "type": "Waste", "turns": ["I have a broken hair dryer and an old laptop. Can I put these in the garbage?"]},
    {"id": 12, "type": "Waste", "turns": ["What should I do with all the fallen leaves and small twigs I raked up from my backyard?"]},
    {"id": 13, "type": "Waste", "turns": ["I am moving out and need to get rid of a large sectional sofa.", "What if I suspect it has bedbugs? Does the process change?"]},
    {"id": 14, "type": "Waste", "turns": ["I found some medical needles in my basement from a previous tenant. How do I toss them?", "Can I just put them in a plastic water bottle and throw that in the trash?"]},
    {"id": 15, "type": "Waste", "turns": ["Where does my Tim Hortons coffee cup go?", "What about the plastic lid and the cardboard sleeve?"]}
]

def get_llm_score(metric_name, context, question, answer):
    """
    Uses the Qwen model to act as a judge and score the RAG performance.
    """
    from modules.llm_interface import call_qwen_model
    
    prompt = f"""
    You are an AI Quality Auditor. Rate the following RAG interaction on a scale of 0.0 to 1.0 for the metric: {metric_name}.
    
    CONTEXT: {context}
    QUESTION: {question}
    ANSWER: {answer}
    
    Definition of {metric_name}:
    - Faithfulness: Is the answer derived ONLY from the context? (1.0 = Yes, 0.0 = Hallucination)
    - Relevance: Does the answer directly address the user's need? (1.0 = Perfect, 0.0 = Irrelevant)
    - Precision: Was the retrieved context actually useful for this question? (1.0 = Highly useful, 0.0 = Useless)
    
    Provide ONLY a numeric score between 0.0 and 1.0.
    """
    messages = [{"role": "system", "content": "You are a metrics evaluator. Output numbers only."},
                {"role": "user", "content": prompt}]
    
    try:
        score_raw = call_qwen_model(messages)
        return float(score_raw.strip())
    except:
        return 0.5

def run_evaluation():
    db = load_brain()
    results = []

    print(f"🚀 Starting Evaluation of {len(TEST_CASES)} cases...")

    for case in TEST_CASES:
        history = []
        last_response = ""
        context_used = ""
        
        print(f"Evaluating Case #{case['id']} ({case['type']})...")

        for turn in case['turns']:
            context, objective = get_city_context(turn, db)
            context_used = context
            last_response = process_query(turn, context, objective, history)
            history.append({"role": "user", "content": turn})
            history.append({"role": "assistant", "content": last_response})

        # Calculate Industry Metrics
        faithfulness = get_llm_score("Faithfulness", context_used, case['turns'][-1], last_response)
        relevance = get_llm_score("Relevance", context_used, case['turns'][-1], last_response)
        precision = get_llm_score("Precision", context_used, case['turns'][-1], last_response)

        results.append({
            "ID": case['id'],
            "Category": case['type'],
            "Query": case['turns'][-1],
            "Faithfulness": faithfulness,
            "Answer_Relevance": relevance,
            "Context_Precision": precision,
            "Overall_Success": 1 if (faithfulness + relevance) / 2 > 0.7 else 0
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(EVAL_RESULTS_FILE, index=False)
    
    print("\n--- FINAL METRICS SUMMARY ---")
    print(df[["Faithfulness", "Answer_Relevance", "Context_Precision"]].mean())
    print(f"\n✅ Evaluation complete. Results saved to {EVAL_RESULTS_FILE}")

if __name__ == "__main__":
    run_evaluation()