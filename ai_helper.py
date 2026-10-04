import json
from groq import Groq
import streamlit as st


def get_groq_client() -> Groq:
    """Initializes the Groq SDK client using secrets."""
    api_key = st.secrets["groq"]["api_key"]
    return Groq(api_key=api_key)


def evaluate_explanation_open_source(
    source_notes: str, student_text: str
) -> dict:
    """Evaluates student explanation against study notes using Llama 3 via Groq.

    Returns a structured dictionary containing feedback and identified gaps.
    """
    client = get_groq_client()

    system_prompt = f"""
    You are an expert Socratic tutor evaluating a student who is using the Feynman teach-back technique.
    
    SOURCE MATERIAL / NOTES:
    \"\"\"{source_notes}\"\"\"

    Analyze the student's explanation against the source material.
    You MUST respond STRICTLY in valid JSON format with the following keys:
    1. "feedback": A encouraging response that highlights accurate insights, points out misconceptions, and asks ONE probing follow-up question.
    2. "gaps": A list of short strings (2-4 words each) representing important concepts from the source material that were missing, incomplete, or wrong in the explanation. Return an empty array [] if no gaps exist.
    3. "is_accurate": A boolean (true/false) indicating if their overall core understanding is sound.

    Example JSON structure:
    {{
        "feedback": "Great job covering the basic idea! However, what happens during the actual transfer phase?",
        "gaps": ["Transfer Phase Dynamics"],
        "is_accurate": false
    }}
    """

    user_message = f"Here is my explanation of what I learned:\n{student_text}"

    try:
        # Call open-weight Llama 3 model via Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Open-weight SOTA model
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
        )

        raw_json = response.choices[0].message.content
        parsed_data = json.loads(raw_json)
        return parsed_data

    except Exception as e:
        # Fallback dictionary if JSON parsing or API call encounters an issue
        return {
            "feedback": f"Thanks for sharing! What is the next key concept you'd like to dive into?",
            "gaps": [],
            "is_accurate": True,
            "error": str(e),
        }