from ai_helper import evaluate_explanation_open_source
from db_helper import fetch_session_gaps, save_knowledge_gap, save_message


def process_student_input(
    session_id: str, source_notes: str, student_text: str
) -> dict:
    """Master pipeline:

    1. Saves student input message to Snowflake.
    2. Calls Open-Weight Llama model via Groq to evaluate explanation.
    3. Saves AI response to Snowflake.
    4. Persists any newly identified knowledge gaps to Snowflake.
    5. Returns evaluation dictionary to Frontend.
    """
    # 1. Store student's message
    save_message(session_id=session_id, sender="USER", content=student_text)

    # 2. Run open-source AI evaluation
    eval_result = evaluate_explanation_open_source(
        source_notes=source_notes, student_text=student_text
    )

    ai_feedback = eval_result.get("feedback", "")
    gaps_found = eval_result.get("gaps", [])

    # 3. Store AI feedback response
    if ai_feedback:
        save_message(session_id=session_id, sender="AI", content=ai_feedback)

    # 4. Save any identified knowledge gaps
    for gap in gaps_found:
        if gap and isinstance(gap, str):
            save_knowledge_gap(session_id=session_id, concept=gap)

    # 5. Fetch updated list of active session gaps for sidebar update
    active_gaps = fetch_session_gaps(session_id=session_id)

    return {
        "feedback": ai_feedback,
        "gaps": gaps_found,
        "active_gaps": active_gaps,
        "is_accurate": eval_result.get("is_accurate", False),
    }