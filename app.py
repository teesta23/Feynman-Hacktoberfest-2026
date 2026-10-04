"""TeachBack: learn by teaching.

Flow: pick a topic -> teach it (voice or typing) -> answer the question TeachBack
asks -> get scored -> follow-up question or teach more. Stuck? Ask for any term.

AI runs on Snowflake Cortex COMPLETE with an open-weight model (Llama / Mistral).
Without Snowflake secrets the app runs in demo mode with offline replies, so the
whole flow still clicks through. Voice uses open-source Whisper if installed.
"""
import html
import json
import os
import re
import tempfile
import uuid
from collections import Counter

import streamlit as st

try:
    import whisper  # pip install openai-whisper (needs ffmpeg)
except ImportError:
    whisper = None

try:
    import snowflake.connector
except ImportError:
    snowflake = None

st.set_page_config(
    page_title="TeachBack",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# stage: "topic" -> "teach" -> "answer" -> "next" (-> "answer" or "teach" ...)
DEFAULTS = {
    "messages": [],        # {"role": "user"|"ai", "kind": "text"|"question"|"feedback"|"explain", "content": str}
    "topic": "",
    "stage": "topic",
    "question": "",
    "follow_up": "",
    "explanation": "",     # everything the user has taught so far
    "key_terms": [],
    "scores": [],
    "draft": "",           # voice transcript waiting for review
    "mic_n": 0,            # bump to reset the audio widget
    "pending": None,       # text waiting to be processed by the AI
    "pending_term": None,  # term waiting to be explained
    "llm_error": "",
    "session_id": str(uuid.uuid4()),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_chat():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, list) else []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.mic_n += 1


def mastery():
    s = st.session_state.scores
    return round(sum(s) / len(s)) if s else 0


# ---------------------------------------------------------------------------
# Snowflake Cortex (open-weight models) with offline fallback
# ---------------------------------------------------------------------------
def _secret_section(name):
    try:
        return dict(st.secrets[name])
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def get_conn():
    cfg = _secret_section("snowflake")
    if snowflake is None or not cfg:
        return None, "no Snowflake secrets"
    try:
        return snowflake.connector.connect(**cfg), ""
    except Exception as e:  # bad creds, network, etc.
        return None, str(e)[:200]


CORTEX_MODEL = _secret_section("cortex").get("model", "llama3.1-70b")


def cortex(prompt):
    """Return model text, or None if Cortex isn't available (caller falls back)."""
    conn, _ = get_conn()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", (CORTEX_MODEL, prompt))
            return cur.fetchone()[0].strip()
        finally:
            cur.close()
    except Exception as e:
        st.session_state.llm_error = str(e)[:200]
        return None


def parse_json(raw):
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


STOP = set(
    """about above after again against because before being below between both could does doing
    during each from further have having here into itself just more most other over same should
    some such than that their theirs them then there these they this those through under until very
    what when where which while whom with would your yours basically actually really thing things
    like kind going gonna means called something someone because which""".split()
)


def guess_terms(text, n=3, skip=""):
    skip_words = set(re.findall(r"[a-z]+", skip.lower()))
    words = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text.lower())
    counts = Counter(w for w in words if w not in STOP and w not in skip_words)
    return [w for w, _ in counts.most_common(n)]


def ai_question(topic, explanation, latest):
    prompt = f"""You are TeachBack, a friendly tutor using the Feynman technique.
A student is teaching you "{topic}". Everything they have explained so far:
\"\"\"{explanation}\"\"\"
Their latest message: \"\"\"{latest}\"\"\"

Find the most important idea they skipped, got wrong, or described without explaining WHY.
Reply with ONLY a JSON object:
{{"reaction": "1-2 warm sentences reacting to what they said (no answers, no lecturing)",
  "question": "ONE conceptual question that makes them explain why/how, tied to what they said",
  "key_terms": ["2-4 key terms for this question that they might need explained"]}}"""
    data = parse_json(cortex(prompt))
    if data and data.get("question"):
        return data
    terms = guess_terms(explanation, skip=topic) or [topic]
    t = terms[0]
    return {
        "reaction": f"Ooh nice, I like how you brought up **{t}**. Let me poke at that a little. 👀",
        "question": f"You mentioned {t}. Why does it matter for {topic}? "
        f"What would go wrong if it wasn't there?",
        "key_terms": terms,
    }


def ai_grade(topic, explanation, question, answer):
    prompt = f"""You are TeachBack, grading how well a student understands "{topic}".
What they taught earlier: \"\"\"{explanation}\"\"\"
Question asked: \"\"\"{question}\"\"\"
Their answer: \"\"\"{answer}\"\"\"

Grade their understanding of the reasoning, not their wording. Reply with ONLY a JSON object:
{{"score": integer 0-100,
  "feedback": "2-3 encouraging sentences: what they got right, then the gap (hint at it, don't fully solve it)",
  "gaps": ["0-3 short concept names they missed or got wrong"],
  "follow_up": "one follow-up question that targets the biggest gap"}}"""
    data = parse_json(cortex(prompt))
    if data and "score" in data:
        try:
            data["score"] = max(0, min(100, int(data["score"])))
            return data
        except (TypeError, ValueError):
            pass
    n_words = len(answer.split())
    has_why = bool(re.search(r"\b(because|so that|which means|therefore|since|so)\b", answer.lower()))
    score = min(95, 35 + min(n_words, 60) // 2 + (15 if has_why else 0))
    gaps = guess_terms(question, 2, skip=topic)
    return {
        "score": score,
        "feedback": (
            "Nice! 🎉 You've got the main idea. "
            + ("I like that you explained the *why*, not just the what. " if has_why else
               "Try adding the *why*: use a 'because...' to connect cause and effect. ")
            + "There's still a small gap in the deeper reasoning, but you're getting there."
        ),
        "gaps": gaps,
        "follow_up": f"Can you give a real-world example of {topic} and walk me through it step by step?",
    }


def ai_explain(topic, term):
    prompt = f"""A student learning "{topic}" is stuck on "{term}".
Explain it simply in markdown, under 120 words:
**{term}:** one-sentence definition in plain words.
**Think of it like:** a simple analogy.
**Why it matters here:** 1-2 sentences linking it to {topic}.
Don't answer any question they're working on for them."""
    out = cortex(prompt)
    if out:
        return out
    return (
        f"**{term}:** (demo mode) a short, plain-words definition would appear here once "
        f"Snowflake Cortex is connected.\n**Think of it like:** a simple everyday analogy.\n"
        f"**Why it matters here:** how {term} fits into {topic}."
    )


def save_attempt(question, answer, result):
    """Log each graded answer to Snowflake so progress survives the session."""
    conn, _ = get_conn()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS TEACHBACK_ATTEMPTS (
                SESSION_ID STRING, TOPIC STRING, EXPLANATION STRING, QUESTION STRING,
                ANSWER STRING, SCORE INT, FEEDBACK STRING, GAPS VARIANT,
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP())"""
        )
        cur.execute(
            """INSERT INTO TEACHBACK_ATTEMPTS
               (SESSION_ID, TOPIC, EXPLANATION, QUESTION, ANSWER, SCORE, FEEDBACK, GAPS)
               SELECT %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s)""",
            (
                st.session_state.session_id, st.session_state.topic, st.session_state.explanation,
                question, answer, result["score"], result.get("feedback", ""),
                json.dumps(result.get("gaps", [])),
            ),
        )
        cur.close()
    except Exception as e:
        st.session_state.llm_error = f"save failed: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# Voice (open-source Whisper)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("base")


def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name
    try:
        return load_whisper().transcribe(path, fp16=False)["text"].strip()
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Flow logic
# ---------------------------------------------------------------------------
STUCK = re.compile(r"^\s*(i\s*(don'?t|do not)\s*know|idk|no idea|not sure|i'?m stuck|help)\b", re.I)


def add(role, content, kind="text"):
    st.session_state.messages.append({"role": role, "kind": kind, "content": content})


def handle(text):
    """Route one user message (typed or spoken) based on the current stage."""
    s = st.session_state
    text = text.strip()
    if not text:
        return

    if s.stage == "topic":
        s.topic = text[:80]
        s.stage = "teach"
        add("ai", f"Ooh, **{s.topic}**! 💗 Teach it to me like I'm your friend who missed class. "
                  "Talk or type, whatever's easier. Don't worry about being perfect.")
        return

    add("user", text)

    if s.stage in ("teach", "next"):
        s.explanation = (s.explanation + "\n" + text).strip()
        q = ai_question(s.topic, s.explanation, text)
        add("ai", q.get("reaction", ""))
        s.question = q["question"]
        s.key_terms = [t for t in q.get("key_terms", []) if t][:4]
        add("ai", s.question, kind="question")
        s.stage = "answer"
        return

    if s.stage == "answer":
        if STUCK.match(text):
            term = s.key_terms[0] if s.key_terms else s.topic
            add("ai", f"Totally okay! Here's a nudge on **{term}**, then give it another go. 💪")
            add("ai", ai_explain(s.topic, term), kind="explain")
            return
        r = ai_grade(s.topic, s.explanation, s.question, text)
        s.scores.append(r["score"])
        save_attempt(s.question, text, r)
        gaps = [g for g in r.get("gaps", []) if g]
        body = f"**{r['score']}/100** · {r.get('feedback', '')}"
        if gaps:
            body += "\n**Worth revisiting:** " + ", ".join(gaps)
        add("ai", body, kind="feedback")
        s.key_terms = (gaps + [t for t in s.key_terms if t not in gaps])[:4]
        s.follow_up = r.get("follow_up", "")
        s.stage = "next"


# Callbacks only queue work; the AI runs in the main script so spinners show in place.
def on_chat_submit():
    st.session_state.pending = st.session_state.get("chat_box", "")


def send_draft():
    st.session_state.pending = st.session_state.get("draft_box", "")
    st.session_state.draft = ""


def discard_draft():
    st.session_state.draft = ""


def ask_term(term):
    st.session_state.pending_term = term


def ask_custom_term():
    term = st.session_state.get("term_box", "").strip()
    if term:
        st.session_state.pending_term = term
    st.session_state.term_box = ""


def go_follow_up():
    s = st.session_state
    s.question = s.follow_up or s.question
    add("ai", s.question, kind="question")
    s.stage = "answer"


def go_teach_more():
    st.session_state.stage = "teach"
    add("ai", "Love it. Keep going, teach me the next part. 📚")


def pick_topic(t):
    st.session_state.pending = t


# ---------------------------------------------------------------------------
# Styles (your pink theme, plus classes for the new pieces)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fredoka:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: "DM Sans", sans-serif; }
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(255, 187, 211, 0.28), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(255, 215, 229, 0.18), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(255, 134, 174, 0.15), transparent 35%),
        linear-gradient(135deg, #7C2348 0%, #A82D59 48%, #8E204B 100%);
    color: white;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(53, 13, 34, 0.97); border-right: 1px solid rgba(255,255,255,0.12); }
[data-testid="stSidebar"] * { color: #FFF7FA; }
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
.block-container { max-width: 1050px; padding-top: 1.5rem; padding-bottom: 8rem; }
.brand { font-family: "Fredoka", sans-serif; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
.tagline { color: #EAB4C8; font-size: 0.98rem; margin-bottom: 25px; }
.hero { text-align: center; padding: 12px 0 22px 0; }
.hero-title { font-family: "Fredoka", sans-serif; font-size: 3.3rem; font-weight: 700; letter-spacing: -1px; color: white; margin: 0; }
.hero-subtitle { font-size: 1.15rem; color: #FFD7E5; margin-top: 4px; }
.welcome { max-width: 790px; margin: 25px auto 20px auto; padding: 30px; border-radius: 28px;
    background: rgba(255,255,255,0.97); color: #432333; box-shadow: 0 18px 45px rgba(40,0,20,0.20); }
.welcome-title { font-family: "Fredoka", sans-serif; color: #A42B57; font-size: 1.65rem; font-weight: 700; margin-bottom: 10px; }
.welcome-text { font-size: 1.08rem; line-height: 1.75; }
.user-row { display: flex; justify-content: flex-end; margin: 20px 0; }
.user-bubble { max-width: 72%; background: #FFD8E7; color: #4B2637; padding: 17px 21px;
    border-radius: 22px 22px 6px 22px; font-size: 1.05rem; line-height: 1.65; box-shadow: 0 8px 24px rgba(45,0,20,0.13); }
.ai-row { display: flex; justify-content: flex-start; margin: 20px 0; }
.ai-avatar { width: 45px; height: 45px; min-width: 45px; border-radius: 50%; background: #FFE4ED;
    display: flex; align-items: center; justify-content: center; font-size: 1.35rem; margin-right: 11px; }
.ai-bubble { max-width: 76%; background: rgba(255,255,255,0.97); color: #432333; padding: 19px 22px;
    border-radius: 6px 22px 22px 22px; font-size: 1.05rem; line-height: 1.7; box-shadow: 0 10px 30px rgba(45,0,20,0.16); }
.ai-name { color: #A42B57; font-size: 0.9rem; font-weight: 800; margin-bottom: 6px; }
.ai-bubble.question { background: #FFF0B8; color: #503C12; }
.ai-bubble.question .ai-name { color: #795900; }
.ai-bubble.question .q-text { font-size: 1.12rem; font-weight: 600; }
.ai-bubble.feedback { background: #DDF5E7; color: #244B34; }
.ai-bubble.feedback .ai-name { color: #2D7650; }
.ai-bubble.explain { background: #EAF1FF; color: #22325A; }
.ai-bubble.explain .ai-name { color: #3355A8; }
.hint { font-size: 0.92rem; opacity: 0.8; margin-top: 8px; }
.record-title { text-align: center; color: #A42B57; font-family: "Fredoka", sans-serif; font-size: 1.45rem; font-weight: 700; }
.record-subtitle { text-align: center; color: #8B6877; font-size: 0.95rem; margin-top: 3px; }
.wave-container { height: 60px; display: flex; justify-content: center; align-items: center; gap: 5px; margin: 6px 0; }
.wave-bar { width: 6px; border-radius: 20px; background: linear-gradient(180deg, #D94E7C, #98234F); animation: wave 1.15s ease-in-out infinite; }
.wave-bar:nth-child(1) { height: 18px; animation-delay: 0s; }
.wave-bar:nth-child(2) { height: 31px; animation-delay: .08s; }
.wave-bar:nth-child(3) { height: 47px; animation-delay: .16s; }
.wave-bar:nth-child(4) { height: 28px; animation-delay: .24s; }
.wave-bar:nth-child(5) { height: 55px; animation-delay: .32s; }
.wave-bar:nth-child(6) { height: 39px; animation-delay: .40s; }
.wave-bar:nth-child(7) { height: 50px; animation-delay: .48s; }
.wave-bar:nth-child(8) { height: 31px; animation-delay: .56s; }
.wave-bar:nth-child(9) { height: 56px; animation-delay: .64s; }
.wave-bar:nth-child(10) { height: 43px; animation-delay: .72s; }
.wave-bar:nth-child(11) { height: 25px; animation-delay: .80s; }
.wave-bar:nth-child(12) { height: 49px; animation-delay: .88s; }
.wave-bar:nth-child(13) { height: 35px; animation-delay: .96s; }
@keyframes wave { 0%, 100% { transform: scaleY(.45); opacity: .55; } 50% { transform: scaleY(1); opacity: 1; } }
.mastery-score { font-family: "Fredoka", sans-serif; font-size: 2.4rem; font-weight: 700; color: #FFD7E5; line-height: 1; }
/* White cards around real Streamlit widgets (keyed containers) */
.st-key-composer, .st-key-stuck, .st-key-nextbox, .st-key-topicbox {
    background: rgba(255,255,255,0.96); border-radius: 28px; padding: 20px 22px;
    box-shadow: 0 18px 45px rgba(40,0,20,0.20); margin: 18px auto; color: #432333;
}
.st-key-composer *, .st-key-stuck *, .st-key-nextbox *, .st-key-topicbox * { color: #432333; }
.card-label { font-family: "Fredoka", sans-serif; color: #A42B57 !important; font-size: 1.2rem; font-weight: 700; margin-bottom: 4px; }
.stButton > button { border-radius: 15px !important; min-height: 46px !important; font-size: 1rem !important; font-weight: 700 !important; transition: .2s ease; }
.stButton > button:hover { transform: translateY(-2px); }
/* Buttons: light background = dark text, dark background = light text */
.stButton > button { background: #FFE4ED !important; color: #4B2637 !important; border: 1px solid #F1B6CB !important; }
.stButton > button:hover { background: #FFD3E2 !important; color: #3A1A29 !important; }
.stButton > button[kind="primary"] { background: #8E204B !important; color: #FFFFFF !important; border-color: #8E204B !important; }
.stButton > button[kind="primary"]:hover { background: #6E1739 !important; }
[data-testid="stSidebar"] .stButton > button { background: #FFD8E7 !important; color: #4B2637 !important; border-color: #FFD8E7 !important; }
.stButton > button * { color: inherit !important; background: transparent !important; border: none !important; }
/* Inputs: white field, dark text */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: #FFFFFF !important; color: #2E1420 !important; }
[data-testid="stChatInput"] textarea::placeholder, .stTextInput input::placeholder { color: #8B6877 !important; }
[data-testid="stChatInput"] button { background: #8E204B !important; color: #FFFFFF !important; }
[data-testid="stChatInput"] button svg { fill: #FFFFFF !important; color: #FFFFFF !important; }
.stTextInput input, .stTextArea textarea { background: #FFFFFF !important; color: #2E1420 !important; border: 1px solid #F1B6CB !important; }
[data-testid="stChatInput"] textarea { font-size: 1.05rem !important; }
.stTextInput input, .stTextArea textarea { font-size: 1.05rem !important; border-radius: 15px !important; }
@media (max-width: 700px) {
    .hero-title { font-size: 2.5rem; }
    .user-bubble, .ai-bubble { max-width: 88%; }
}
</style>
""",
    unsafe_allow_html=True,
)


def fmt(text):
    """Escape user/model text, then allow **bold**, *italic* and line breaks."""
    t = html.escape(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    return t.replace("\n", "<br>")


def show(html_str):
    # One line, no indentation, so Markdown never turns it into a code block.
    st.markdown(re.sub(r"\n\s*", "", html_str), unsafe_allow_html=True)


WAVE = '<div class="wave-container">' + '<div class="wave-bar"></div>' * 13 + "</div>"
NAMES = {"question": "🤔 Your turn to think", "feedback": "📈 How you did", "explain": "💡 Quick explainer"}


def render_message(m):
    if m["role"] == "user":
        show(f'<div class="user-row"><div class="user-bubble">{fmt(m["content"])}</div></div>')
        return
    kind = m.get("kind", "text")
    body = fmt(m["content"])
    if kind == "question":
        body = f'<div class="q-text">{body}</div>'
    show(
        f'<div class="ai-row"><div class="ai-avatar">🧠</div>'
        f'<div class="ai-bubble {kind}"><div class="ai-name">{NAMES.get(kind, "TeachBack")}</div>{body}</div></div>'
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
conn, conn_err = get_conn()
with st.sidebar:
    show('<div class="brand">🧠 TeachBack</div><div class="tagline">Learn by teaching ✨</div>')
    st.button("＋ New chat", use_container_width=True, on_click=reset_chat)
    st.markdown("---")
    if st.session_state.topic:
        st.markdown(f"### 📚 Learning\n**{html.escape(st.session_state.topic)}**")
    if st.session_state.scores:
        st.markdown("### 📈 Your mastery")
        show(f'<div class="mastery-score">{mastery()}%</div>')
        st.progress(mastery() / 100)
        st.caption(f"{len(st.session_state.scores)} answer(s) checked")
        st.markdown("---")
    st.markdown("### 💡 How TeachBack works")
    st.markdown(
        "**1️⃣ Teach** · explain it in your own words\n\n"
        "**2️⃣ Challenge** · I ask a question about what you said\n\n"
        "**3️⃣ Think** · answer without your notes\n\n"
        "**4️⃣ Diagnose** · I score it and spot the gaps\n\n"
        "**5️⃣ Improve** · stuck? ask me to explain any term"
    )
    st.markdown("---")
    if conn is not None:
        st.caption(f"🟢 AI: Snowflake Cortex · {CORTEX_MODEL}")
    else:
        st.caption("🟡 AI: demo mode (add Snowflake secrets for Cortex)")
    st.caption("🟢 Voice: Whisper" if whisper else "⚪ Voice off · pip install openai-whisper")
    if st.session_state.llm_error:
        st.caption(f"⚠️ Last AI error: {st.session_state.llm_error}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
show(
    '<div class="hero"><div class="hero-title">TeachBack 🧠</div>'
    '<div class="hero-subtitle">Explain it. Get challenged. Actually understand it.</div></div>'
)

s = st.session_state

if s.stage == "topic" and not s.messages:
    show(
        '<div class="welcome"><div class="welcome-title">Hey! 👋 Let\'s learn something.</div>'
        '<div class="welcome-text">TeachBack helps you learn by teaching. Pick a topic and explain it like '
        "you're teaching your friend. I'll listen, ask you a conceptual question, and help you discover any "
        "gaps in your understanding.<br><br><b>You don't have to be perfect. Just explain it how you "
        "understand it.</b> ✨</div></div>"
    )
    with st.container(key="topicbox"):
        show('<div class="card-label">📚 What are you learning today?</div>')
        st.caption("Type it in the box at the bottom, or pick one:")
        cols = st.columns(4)
        for col, t in zip(cols, ["Recursion", "Photosynthesis", "Newton's laws", "Supply and demand"]):
            col.button(t, use_container_width=True, on_click=pick_topic, args=(t,))

for m in s.messages:
    render_message(m)

# Run queued AI work right here, under the chat, so the spinner sits where the reply will appear.
if s.pending is not None:
    text, s.pending = s.pending, None
    with st.spinner("🧠 TeachBack is thinking..."):
        handle(text)
    st.rerun()
if s.pending_term:
    term, s.pending_term = s.pending_term, None
    add("user", f"Can you explain **{term}**?")
    with st.spinner(f"💡 Explaining {term}..."):
        add("ai", ai_explain(s.topic, term), kind="explain")
    st.rerun()

# Contextual action area: exactly one mic, whose meaning follows the stage.
if s.stage in ("teach", "answer"):
    with st.container(key="composer"):
        if s.stage == "teach":
            title, sub = f"🎙️ Teach me {html.escape(s.topic)}", "Explain it in your own words. Pauses are okay!"
        else:
            title, sub = "🎙️ Answer TeachBack", "Explain your reasoning out loud. Take your time."
        show(f'<div class="record-title">{title}</div><div class="record-subtitle">{sub}</div>{WAVE}')

        if s.draft:
            st.markdown("**📝 I heard you say...** (fix anything I misheard)")
            st.text_area("Transcript", value=s.draft, key="draft_box", height=140, label_visibility="collapsed")
            c1, c2 = st.columns(2)
            c1.button("✨ Send", type="primary", use_container_width=True, on_click=send_draft)
            c2.button("🎙️ Record again", use_container_width=True, on_click=discard_draft)
        elif whisper is not None:
            audio = st.audio_input("Record", key=f"mic_{s.mic_n}", label_visibility="collapsed")
            if audio is not None:
                with st.spinner("✨ Listening to you..."):
                    try:
                        text = transcribe(audio.getvalue())
                    except Exception as e:
                        text = ""
                        st.error(f"Couldn't transcribe that ({e}). Try again or type below.")
                s.mic_n += 1  # fresh widget next run, so the clip is never processed twice
                if text:
                    s.draft = text
                    st.rerun()
                else:
                    st.warning("I didn't catch anything. Try again, or type below. 💗")
        else:
            st.caption("Voice is off on this machine. Type your explanation in the box below. ⌨️")

if s.stage == "answer":
    with st.container(key="stuck"):
        show('<div class="card-label">😵 Stuck? Ask me about a term</div>')
        if s.key_terms:
            cols = st.columns(len(s.key_terms))
            for i, (col, t) in enumerate(zip(cols, s.key_terms)):
                col.button(f"💡 {t}", key=f"term_{i}_{t}", use_container_width=True, on_click=ask_term, args=(t,))
        st.text_input(
            "Explain a term", key="term_box", placeholder="Type any word you're unsure about and press Enter",
            label_visibility="collapsed", on_change=ask_custom_term,
        )

if s.stage == "next":
    with st.container(key="nextbox"):
        show('<div class="card-label">What next?</div>')
        c1, c2 = st.columns(2)
        c1.button("🔁 Try a follow-up question", type="primary", use_container_width=True, on_click=go_follow_up)
        c2.button("🧠 Teach me more", use_container_width=True, on_click=go_teach_more)
        if s.key_terms:
            st.caption("Or brush up on a gap:")
            cols = st.columns(len(s.key_terms))
            for i, (col, t) in enumerate(zip(cols, s.key_terms)):
                col.button(f"💡 {t}", key=f"gap_{i}_{t}", use_container_width=True, on_click=ask_term, args=(t,))

PLACEHOLDERS = {
    "topic": "What do you want to learn? e.g. recursion",
    "teach": "Or type your explanation here...",
    "answer": "Or type your answer here... (say \"I don't know\" for a hint)",
    "next": "Keep teaching, or type anything to continue...",
}
st.chat_input(PLACEHOLDERS[s.stage], key="chat_box", on_submit=on_chat_submit)
