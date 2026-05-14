import openai
import pandas as pd
from dotenv import load_dotenv
import os
import json
from typing import Dict

load_dotenv()

client = openai.OpenAI()

def detect_hallucination(prompt: str, response: str, ground_truth: str = None) -> Dict:
    """
    Uses GPT-4o-mini as a judge to detect hallucinations.
    Returns detailed scoring.
    """
    judgment_prompt = f"""
    You are an expert fact-checker. Analyze if the response contains any hallucinated information.

    Prompt: {prompt}
    Response: {response}
    {f"Ground Truth: {ground_truth}" if ground_truth else ""}

    Rate from 0 to 1:
    - hallucination_score (0 = no hallucination, 1 = completely made up)
    - confidence (how confident you are in your judgment)

    Return ONLY valid JSON:
    {{"hallucination_score": 0.XX, "confidence": 0.XX, "reason": "short explanation"}}
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judgment_prompt}],
            temperature=0.0,
            max_tokens=200
        )
        
        result = json.loads(completion.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        return {"hallucination_score": 0.5, "confidence": 0.5, "reason": f"Error: {str(e)}"}


# Example batch processing
def analyze_batch(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    
    for _, row in df.iterrows():
        result = detect_hallucination(
            prompt=row['prompt'],
            response=row['response'],
            ground_truth=row.get('ground_truth')
        )
        results.append(result)
    
    df_results = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), df_results], axis=1)
    
    # Overall metrics
    df['is_hallucination'] = df['hallucination_score'] > 0.6
    print(f"Overall Hallucination Rate: {df['is_hallucination'].mean()*100:.1f}%")
    
    return df
