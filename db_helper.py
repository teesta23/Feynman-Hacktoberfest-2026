import uuid
import pandas as pd
import snowflake.connector
import streamlit as st


def get_snowflake_connection():
    """Initializes a Snowflake connection using credentials from Streamlit secrets."""
    sf_config = st.secrets["snowflake"]
    return snowflake.connector.connect(
        account=sf_config["account"],
        user=sf_config["user"],
        password=sf_config["password"],
        warehouse=sf_config["warehouse"],
        database=sf_config["database"],
        schema=sf_config["schema"],
    )


def save_message(session_id: str, sender: str, content: str):
    """Saves a single conversation turn into the Snowflake MESSAGES table."""
    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            msg_id = str(uuid.uuid4())
            sql = """
                INSERT INTO MESSAGES (MESSAGE_ID, SESSION_ID, SENDER, CONTENT)
                VALUES (%s, %s, %s, %s)
            """
            cur.execute(sql, (msg_id, session_id, sender, content))
        conn.commit()
    finally:
        conn.close()


def save_knowledge_gap(session_id: str, concept: str):
    """Saves an identified knowledge gap into the KNOWLEDGE_GAPS table."""
    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            gap_id = str(uuid.uuid4())
            sql = """
                INSERT INTO KNOWLEDGE_GAPS (GAP_ID, SESSION_ID, CONCEPT, STATUS)
                VALUES (%s, %s, %s, 'UNRESOLVED')
            """
            cur.execute(sql, (gap_id, session_id, concept))
        conn.commit()
    finally:
        conn.close()


def fetch_session_gaps(session_id: str) -> list:
    """Retrieves all unresolved knowledge gaps for a given study session."""
    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT GAP_ID, CONCEPT 
                FROM KNOWLEDGE_GAPS 
                WHERE SESSION_ID = %s AND STATUS = 'UNRESOLVED'
                ORDER BY CREATED_AT DESC
            """
            cur.execute(sql, (session_id,))
            rows = cur.fetchall()
            return [{"gap_id": row[0], "concept": row[1]} for row in rows]
    finally:
        conn.close()