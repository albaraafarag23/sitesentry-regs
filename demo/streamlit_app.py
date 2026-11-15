"""Public demo: the chat page as a Streamlit app.

Why this exists: free hosts for a Docker/FastAPI server ask for a payment card.
Streamlit Community Cloud is free with only a GitHub login, but it runs
Streamlit apps. So this file is a second, thin front end. It calls the *same*
agent, tools, citations and database code as the FastAPI server; nothing about
retrieval or answering is reimplemented here.

Run locally:
    streamlit run demo/streamlit_app.py
"""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

# --- Settings must be in place before the app package is imported -------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # so "import app" works from demo/
os.environ.setdefault("EMBEDDING_BACKEND", "onnx")  # no PyTorch on the demo server
try:
    for key, value in st.secrets.items():           # Streamlit Cloud: secrets typed into its dashboard
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass                                            # running locally: values come from .env instead

from app import db, limits  # noqa: E402
from app.agent import for_memory, run_agent  # noqa: E402
from app.answer import LLMError  # noqa: E402
from app.config import HISTORY_LIMIT  # noqa: E402

MAX_QUESTIONS_PER_VISIT = 15    # the demo runs on a free LLM quota; keep one visitor from using it all
EXAMPLES = [
    "How high must a guard rail be?",
    "SiteSentry flagged no_harness near a roof edge. Which rule does that break?",
    "متى يجب فحص معدات العمل في الأماكن المرتفعة؟",
    "Who prepares the construction safety and health plan?",
]

st.set_page_config(page_title="SiteSentry Regs", page_icon="🦺", layout="centered")
# unicode-bidi: plaintext makes each paragraph pick its own direction, so Arabic reads right-to-left.
st.markdown(
    "<style>[data-testid='stChatMessage'] p, [data-testid='stChatMessage'] li "
    "{unicode-bidi: plaintext; text-align: start;}</style>",
    unsafe_allow_html=True,
)
st.title("SiteSentry Regs")
st.caption(
    "Ask about UAE construction-safety rules in English or Arabic. Every answer cites the clause it came from. "
    "[Source code](https://github.com/albaraafarag23/sitesentry-regs)"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex   # one conversation per browser tab
    st.session_state.turns = []                      # what to show on screen


@st.cache_resource
def prepare_database() -> bool:
    """Runs once per server start: make sure the tables exist."""
    db.ensure_schema()
    return True


def show_details(turn: dict) -> None:
    """Under an answer: which tools the agent used, and the cited clauses."""
    if turn.get("steps"):
        calls = [f"{s['tool']}({', '.join(str(v) for v in s['arguments'].values())})" for s in turn["steps"]]
        st.caption("Tools used: " + " → ".join(calls))
    for source in turn.get("citations", []):
        label = (f"[{source['number']}] {source['document']}, clause {source['clause']} "
                 f"“{source['heading']}”, page {source['page']} ({source['language']})")
        with st.expander(label):
            st.write(source["content"])


def answer(question: str) -> dict:
    """Run the agent with this conversation's memory, save the turn, return what to display."""
    with db.connect() as conn:
        history = db.get_history(conn, st.session_state.session_id, HISTORY_LIMIT)
    engine = st.session_state.get("engine", "Hand-written agent")
    if engine == "LangChain agent":
        from app.agent_langchain import run_agent_langchain

        result = run_agent_langchain(question, history)
    else:
        result = run_agent(question, history)
    with db.connect() as conn:
        db.add_message(conn, st.session_state.session_id, "user", question)
        db.add_message(conn, st.session_state.session_id, "assistant",
                       for_memory(result.answer, result.sources, result.cited))
    citations = [{"number": n, **result.sources[n - 1]} for n in result.cited]
    return {"role": "assistant", "text": result.answer, "steps": result.steps, "citations": citations}


st.radio("Agent implementation", ["Hand-written agent", "LangChain agent"], key="engine", horizontal=True,
         help="Two implementations of the same agent: a hand-written tool-calling loop, and LangChain/LangGraph.")

if not st.session_state.turns and "pending" not in st.session_state:   # a fresh page: offer examples
    st.markdown("**Try one of these**")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()                              # redraw without the examples, then answer below
    st.caption("Documents: OSHJ-CoP-04 Working at Height Safety (English, Arabic) and "
               "OSHJ-CoP-02 OSH Management in Construction (English), Sharjah Prevention and Safety Authority.")

try:
    prepare_database()
except Exception as error:
    st.error(f"The database is not reachable: {type(error).__name__}")
    st.stop()

for turn in st.session_state.turns:                 # redraw the conversation so far
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        show_details(turn)

question = st.chat_input("Ask a question / اكتب سؤالك") or st.session_state.pop("pending", None)
if question:
    asked = sum(1 for turn in st.session_state.turns if turn["role"] == "user")
    st.session_state.turns.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        busy = None
        if asked < MAX_QUESTIONS_PER_VISIT:
            # every visitor together, counted in the database: reloading the page doesn't reset it
            with db.connect() as conn:
                if db.count_questions(conn, 3600) >= limits.PER_HOUR:
                    busy = "The demo has had many questions in the last hour. Please try again later."
                elif db.count_questions(conn, 86400) >= limits.PER_DAY:
                    busy = "The demo has reached its daily limit of questions. Please try again tomorrow."
        if asked >= MAX_QUESTIONS_PER_VISIT:
            reply = {"role": "assistant", "text": "This demo allows a limited number of questions per visit. "
                                                  "Reload the page to start a new conversation."}
        elif busy:
            reply = {"role": "assistant", "text": busy}
        else:
            try:
                with st.spinner("Searching the regulations…"):
                    reply = answer(question)
            except LLMError as error:
                reply = {"role": "assistant", "text": f"The language model is unavailable right now: {error}"}
            except Exception as error:
                reply = {"role": "assistant", "text": f"Something went wrong: {type(error).__name__}"}
        st.markdown(reply["text"])
        show_details(reply)
    st.session_state.turns.append(reply)
