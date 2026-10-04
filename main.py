import os
import json
import uuid

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import snowflake.connector


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")


# --------------------------------------------------
# Snowflake connection
# --------------------------------------------------

def get_snowflake_connection():

    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )


# --------------------------------------------------
# Ask Snowflake Cortex
# --------------------------------------------------

def query_cortex(prompt):

    connection = get_snowflake_connection()

    try:

        cursor = connection.cursor()

        query = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)
        """

        cursor.execute(
            query,
            ("llama3.1-70b", prompt)
        )

        result = cursor.fetchone()

        return result[0] if result else ""

    finally:

        connection.close()


# --------------------------------------------------
# Generate TeachBack response
# --------------------------------------------------

def generate_ai_response(
    grade,
    subject,
    topic,
    messages
):

    conversation = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in messages
    )

    prompt = f"""
You are TeachBack AI.

You are a curious study partner using the
Feynman teach-back method.

The student is learning:

Grade: {grade}
Subject: {subject}
Topic: {topic}

Your job is NOT to immediately teach the student.

Instead:

1. Listen carefully to their explanation.
2. Identify one important missing connection,
   misconception, or weak point.
3. Ask ONE short question that tests that point.
4. Do not reveal the answer.
5. Keep the question appropriate for the student's
   grade and topic.
6. Encourage reasoning rather than memorization.
7. Never ask multiple questions at once.

If the student clearly says they do not know,
briefly explain the specific concept they are
missing and then ask them to teach that concept
back to you.

Conversation:

{conversation}

Respond naturally as a supportive study partner.
Keep the response under 3 sentences.
"""

    return query_cortex(prompt).strip()


# --------------------------------------------------
# Evaluate session
# --------------------------------------------------

def evaluate_session(
    grade,
    subject,
    topic,
    messages
):

    transcript = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in messages
    )

    prompt = f"""
You are an educational evaluator.

Evaluate this teach-back session.

Grade: {grade}
Subject: {subject}
Topic: {topic}

Student/AI dialogue:

{transcript}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "understanding_score": 0,
    "understanding_level": "Beginner",
    "strengths": [],
    "misconceptions_or_gaps": [],
    "syllabus_coverage": "",
    "pedagogical_advice": ""
}}

Rules:

understanding_score must be between 1 and 100.

understanding_level must be one of:

Beginner
Intermediate
Advanced

Identify real strengths and gaps from the dialogue.

Do not invent knowledge that the student did not demonstrate.

Keep pedagogical_advice short and actionable.
"""

    raw = query_cortex(prompt)

    cleaned = raw.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]

    if cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:

        return {
            "understanding_score": 0,
            "understanding_level": "Unknown",
            "strengths": [],
            "misconceptions_or_gaps": [
                "Unable to parse evaluation."
            ],
            "syllabus_coverage": "",
            "pedagogical_advice": cleaned
        }


# --------------------------------------------------
# Store session in Snowflake
# --------------------------------------------------

def store_session(
    grade,
    subject,
    topic,
    messages,
    evaluation
):

    session_id = str(uuid.uuid4())

    connection = get_snowflake_connection()

    try:

        cursor = connection.cursor()

        query = """
        INSERT INTO LEARNING_SESSIONS
        (
            SESSION_ID,
            STUDENT_GRADE,
            SUBJECT,
            TOPIC,
            SYLLABUS_SCOPE,
            TRANSCRIPT,
            EVALUATION_REPORT
        )

        SELECT
            %s,
            %s,
            %s,
            %s,
            %s,
            PARSE_JSON(%s),
            PARSE_JSON(%s)
        """

        cursor.execute(
            query,
            (
                session_id,
                grade,
                subject,
                topic,
                topic,
                json.dumps(messages),
                json.dumps(evaluation)
            )
        )

        connection.commit()

    finally:

        connection.close()

    return session_id


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.json

    grade = data.get("grade", "")
    subject = data.get("subject", "")
    topic = data.get("topic", "")
    messages = data.get("messages", [])

    try:

        reply = generate_ai_response(
            grade,
            subject,
            topic,
            messages
        )

        return jsonify({
            "success": True,
            "reply": reply
        })

    except Exception as error:

        print(error)

        return jsonify({
            "success": False,
            "error": "Unable to contact the AI."
        }), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():

    data = request.json

    grade = data.get("grade", "")
    subject = data.get("subject", "")
    topic = data.get("topic", "")
    messages = data.get("messages", [])

    try:

        evaluation = evaluate_session(
            grade,
            subject,
            topic,
            messages
        )

        session_id = store_session(
            grade,
            subject,
            topic,
            messages,
            evaluation
        )

        return jsonify({
            "success": True,
            "session_id": session_id,
            "evaluation": evaluation
        })

    except Exception as error:

        print(error)

        return jsonify({
            "success": False,
            "error": "Unable to evaluate session."
        }), 500


# --------------------------------------------------
# Start application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )