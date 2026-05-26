import os
import requests
import streamlit as st
import random
import json
import streamlit.components.v1 as components
from urllib.parse import quote
import pandas as pd
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Lucid", layout="wide")
st.title("Lucid — PDF Q&A")

# Preloaded demo documents. Feature 3 will move this to a constants module
# and resolve `doc_name` against actual Qdrant payloads.
PRELOADED_DOCS = [
    {
        "key": "rl",
        "title": "Reinforcement Learning",
        "subtitle": "Sutton & Barto — RL textbook",
        "doc_name": "RLbook2020 2.pdf",
        "accent": "#6366F1",
        "example_questions": [
            "In the k-armed bandit setting, what does an action-value estimate represent?",
            "How can optimistic initial values encourage exploration in action-value methods?",
            "Compare epsilon-greedy action selection with upper-confidence-bound action selection.",
        ],
    },
    {
        "key": "os",
        "title": "Operating Systems",
        "subtitle": "OSTEP — three easy pieces",
        "doc_name": "operating_systems_three_easy_pieces 2.pdf",
        "accent": "#10B981",
        "example_questions": [
            "Why is the process abstraction important in an operating system?",
            "What problem does limited direct execution solve?",
            "Compare turnaround time and response time as scheduling metrics.",
        ],
    },
    {
        "key": "csc263",
        "title": "CSC263",
        "subtitle": "Data structures & analysis",
        "doc_name": "csc263_notes 2.pdf",
        "accent": "#F59E0B",
        "example_questions": [
            "What does asymptotic analysis focus on when measuring running time?",
            "What invariant does a binary heap maintain?",
            "What is the difference between an abstract data type and a data structure?",
        ],
    },
    {
        "key": "mat102",
        "title": "MAT102",
        "subtitle": "Foundations of mathematics",
        "doc_name": "MAT102-Notes-2017-Version1 copy 2.pdf",
        "accent": "#EC4899",
        "example_questions": [
            "What is a mathematical statement?",
            "Compare an implication with its converse.",
            "Compare proof by contrapositive and proof by contradiction.",
        ],
    },
]

# -------------------------
# Session state
# -------------------------
DEFAULT_SESSION_STATE = {
    "is_thinking": False,
    "thinking_message": None,
    "pending_question": None,
    "pending_tag_filter": None,
    "pending_doc_filter": None,
    "messages": [],
    "last_hits": [],
    "last_original_query": None,
    "last_rewritten_query": None,
    "last_confidence": None,
    "uploader_key": 0,
    "flash_success": None,
    "flash_error": None,
    "streaming_enabled": True,
    "rewrite_enabled": True,
    "pending_rewrite_enabled": None,
    "current_page": "chat",
    "flashcards": [],
    "flashcards_doc": None,
    "flashcards_loading": False,
    "flashcards_error": None,
    "session_id": None,
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if not st.session_state.session_id:
    st.session_state.session_id = uuid.uuid4().hex

# -------------------------
# Helpers
# -------------------------

MAX_HISTORY_PAIRS = 3


def get_session_id() -> str:
    """
    Return the anonymous browser-session ID used for private uploads.

    Returns:
        Stable session ID for this Streamlit browser session.
    """
    return st.session_state.session_id


def build_pdf_url(doc_name: str, page: int | None = None) -> str:
    """
    Build a citation URL that includes the anonymous session ID.

    Args:
        doc_name: User-facing document name.
        page: Optional PDF page number.

    Returns:
        Backend PDF URL authorized for this browser session.
    """
    pdf_url = (
        f"{BACKEND_URL}/pdfs/{quote(doc_name, safe='')}"
        f"?session_id={quote(get_session_id(), safe='')}"
    )
    if page:
        pdf_url = f"{pdf_url}#page={page}"
    return pdf_url


components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;
        const root = doc.documentElement;

        function updateSidebarWidth() {
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            const width = sidebar.getBoundingClientRect().width;
            root.style.setProperty('--sb-sidebar-width', width + 'px');
        }

        updateSidebarWidth();

        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar && window.ResizeObserver) {
            if (window.parent._sbSidebarObserver) {
                window.parent._sbSidebarObserver.disconnect();
            }
            const ro = new ResizeObserver(updateSidebarWidth);
            ro.observe(sidebar);
            window.parent._sbSidebarObserver = ro;
        }

        window.addEventListener('resize', updateSidebarWidth);
    })();
    </script>
    """,
    height=0,
)

def build_history_payload(messages: list[dict], exclude_last: bool = True) -> list[dict]:
    """
    Build a bounded chat history payload from session messages.

    The current user question has already been appended to messages by the
    submit handler, so we exclude the last entry by default.
    """
    prior = messages[:-1] if exclude_last and messages else messages
    bounded = prior[-(MAX_HISTORY_PAIRS * 2):]
    return [
        {"role": m["role"], "content": m.get("content", "")}
        for m in bounded
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]


SPINNER_MESSAGES = [
    "Searching your PDFs...",
    "Finding relevant chunks...",
    "Reranking evidence...",
    "Reading citations...",
    "Generating answer...",
]


def get_preloaded_doc_by_name(doc_name: str | None) -> dict | None:
    """
    Look up a preloaded demo document by Qdrant document name.

    Args:
        doc_name: Document name stored in Qdrant.

    Returns:
        Matching document metadata, or None.
    """
    if not doc_name:
        return None
    for d in PRELOADED_DOCS:
        if d["doc_name"] == doc_name:
            return d
    return None


def shorten_label(name: str, max_len: int = 24) -> str:
    """
    Shorten a long label for compact selectboxes and buttons.

    Args:
        name: Label text to display.
        max_len: Maximum display length including ellipsis.

    Returns:
        Original or shortened label.
    """
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def submit_question_from_chip(question: str, doc_name: str) -> None:
    """
    Submit a one-click example question end-to-end.

    Appends the question to the chat thread, sets up the pending-question
    state so the next Streamlit rerun routes it through the streaming or
    non-streaming submit handler, and triggers an immediate rerun.

    Args:
        question: Pre-canned question text to send to the backend.
        doc_name: Document filter to scope retrieval to (matches a
            preloaded demo document filename in PRELOADED_DOCS).
    """
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "citations": [],
        "hits": [],
    })
    st.session_state.pending_question = question
    st.session_state.pending_tag_filter = "All"
    st.session_state.pending_doc_filter = doc_name
    st.session_state.pending_rewrite_enabled = st.session_state.rewrite_enabled
    st.session_state.doc_filter_select = doc_name

    st.session_state.is_thinking = True
    st.session_state.thinking_message = random.choice(SPINNER_MESSAGES)

    st.rerun()


def render_example_questions(doc: dict, location: str) -> None:
    """Render 3 clickable example questions for a preloaded doc.

    location is part of the button key namespace ('landing' or 'chat')
    so the same questions can render in both places in the same script run.
    """
    questions = doc.get("example_questions", [])
    if not questions:
        return

    st.markdown(
        f'<div style="font-size: 0.8rem; opacity: 0.65; margin: 0.75rem 0 0.4rem 0;">'
        f'Try a question about <b>{doc["title"]}</b>:</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(questions))
    for i, (col, q) in enumerate(zip(cols, questions)):
        with col:
            if st.button(q, key=f"exq_{location}_{doc['key']}_{i}", use_container_width=True):
                submit_question_from_chip(q, doc["doc_name"])


@st.cache_data(ttl=10)
def fetch_filter_options(session_id: str) -> dict:
    """
    Fetch available tag, document, and source-type filters from the backend.

    Args:
        session_id: Anonymous browser-session ID for private upload visibility.

    Returns:
        Filter option payload, or empty option lists when the backend is
        unavailable.
    """
    try:
        r = requests.get(
            f"{BACKEND_URL}/filter_options",
            params={"session_id": session_id},
            timeout=15,
        )
        if r.ok:
            return r.json()
    except Exception:
        pass

    return {
        "tags": [],
        "documents": [],
        "source_types": [],
    }


def normalize_tags(raw_tags: list[dict] | list[str] | None) -> list[dict[str, str]]:
    """
    Normalize tag payloads from backend into:
    [{"name": ..., "color": ...}, ...]
    """
    normalized = []
    seen = set()

    for item in raw_tags or []:
        if isinstance(item, dict):
            name = item.get("name")
            color = item.get("color", "#2563EB")
        else:
            name = str(item)
            color = "#2563EB"

        if not name or name in seen:
            continue

        seen.add(name)
        normalized.append({"name": name, "color": color})

    return normalized


def tag_chip(tag_name: str, tag_color: str) -> None:
    """
    Render a compact colored tag chip.

    Args:
        tag_name: Tag label to show.
        tag_color: Hex color used for the chip border and background tint.
    """
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:0.18rem 0.55rem;
            border-radius:999px;
            background:{tag_color}22;
            border:1px solid {tag_color};
            color:white;
            font-size:0.85rem;
            margin-top:0.1rem;
            margin-bottom:0.2rem;
        ">
            {tag_name}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_score(value, digits: int = 3) -> str:
    """
    Format a numeric score for display.

    Args:
        value: Score value from backend.
        digits: Number of decimal places.

    Returns:
        Readable score string.
    """
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def confidence_badge(confidence: dict | None) -> None:
    """
    Render a small confidence badge in the Streamlit UI.
    """
    if not confidence:
        return

    label = confidence.get("label", "Unknown")
    score = confidence.get("score")

    if label == "High":
        border = "#22C55E"
    elif label == "Medium":
        border = "#F59E0B"
    else:
        border = "#EF4444"

    st.markdown(
        f"""
        <div style="
            padding:0.65rem 0.75rem;
            border-radius:0.75rem;
            border:1px solid {border};
            background:{border}18;
            margin-bottom:0.5rem;
        ">
            <div style="font-size:0.85rem; opacity:0.8;">Answer confidence</div>
            <div style="font-size:1.1rem; font-weight:700;">{label}</div>
            <div style="font-size:0.8rem; opacity:0.75;">Score: {format_score(score, 2)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Compact styling
# -------------------------
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.45rem;
        }

        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }

        .small-section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        
        /* ---------- Citation header strip ---------- */
        .citation-header {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.5rem;
            margin-top: 0.55rem;
            margin-bottom: 0.15rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.35rem;
        }
        .citation-header-icon {
            flex-shrink: 0;
            font-size: 0.95rem;
            line-height: 1;
            opacity: 0.85;
        }

        .citation-header-doc {
            flex: 1;
            min-width: 0;
            font-size: 0.88rem;
            font-weight: 600;
            color: #f3f4f6;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .citation-header-page {
            flex-shrink: 0;
            display: inline-block;
            padding: 0.1rem 0.45rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 500;
            color: #d1d5db;
            font-variant-numeric: tabular-nums;
        }

        .citation-header-tag {
            flex-shrink: 0;
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 500;
            color: #fff;
            max-width: 7rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* ---------- Citation links (clickable doc / page) ---------- */
        a.citation-header-doc,
        a.citation-header-page {
            text-decoration: none;
            cursor: pointer;
            transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
        }

        a.citation-header-doc {
            color: #f3f4f6;
        }

        a.citation-header-doc:hover {
            color: #ffffff;
            text-decoration: underline;
        }

        a.citation-header-page:hover {
            background: rgba(255, 255, 255, 0.18) !important;
            border-color: rgba(255, 255, 255, 0.32) !important;
            color: #ffffff !important;
        }

        /* ---------- Citation card container ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.citation-card-marker) {
            background: rgba(255, 255, 255, 0.025) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 0.5rem !important;
            padding: 0.35rem 0.55rem 0.45rem 0.55rem !important;
            margin-bottom: 0.55rem !important;
            transition: border-color 0.15s ease;
        }

        /* Pull header flush with the card top edge */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.citation-card-marker)
        .citation-header {
            margin-top: 0.1rem !important;
            background: transparent !important;
            border: none !important;
            padding: 0.15rem 0 0.25rem 0 !important;
        }

        /* ---------- Score lines ---------- */
        .citation-score-line {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 0.1rem 0.1rem;
            font-size: 0.78rem;
            font-variant-numeric: tabular-nums;
        }

        .citation-score-line .citation-score-label {
            letter-spacing: 0.02em;
        }

        .citation-score-line-raw {
            color: #9ca3af;
            opacity: 0.78;
        }

        .citation-score-line-raw .citation-score-val {
            font-weight: 500;
        }

        .citation-score-line-norm {
            color: #f3f4f6;
        }

        .citation-score-line-norm .citation-score-val {
            font-weight: 700;
            color: #ffffff;
        }

        /* ---------- Flashcards ---------- */
        .flashcard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .flashcard {
            perspective: 1200px;
            display: block;
            cursor: pointer;
            min-height: 200px;
        }

        .flashcard input.flashcard-toggle {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }

        .flashcard-inner {
            position: relative;
            width: 100%;
            min-height: 200px;
            transition: transform 0.55s cubic-bezier(0.4, 0.0, 0.2, 1);
            transform-style: preserve-3d;
        }

        .flashcard:has(input.flashcard-toggle:checked) .flashcard-inner {
            transform: rotateY(180deg);
        }

        .flashcard-face {
            position: absolute;
            inset: 0;
            backface-visibility: hidden;
            -webkit-backface-visibility: hidden;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 0.7rem;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.03);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        }

        .flashcard-back {
            transform: rotateY(180deg);
            background: rgba(99, 102, 241, 0.07);
            border-color: rgba(99, 102, 241, 0.30);
        }

        .flashcard-index {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.55;
        }

        .flashcard-front-content {
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.5;
            color: #f3f4f6;
            margin: 0.5rem 0;
            flex: 1;
            display: flex;
            align-items: center;
        }

        .flashcard-back-content {
            font-size: 0.95rem;
            line-height: 1.55;
            color: #f3f4f6;
            flex: 1;
            overflow-y: auto;
            margin-bottom: 0.5rem;
        }

        .flashcard-hint {
            font-size: 0.72rem;
            opacity: 0.5;
            text-align: right;
            font-style: italic;
        }

        .flashcard-citation-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            align-items: center;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        .flashcard-citation {
            font-size: 0.75rem;
            color: #d1d5db;
            text-decoration: none;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            transition: background 0.12s ease, color 0.12s ease;
        }

        .flashcard-citation:hover {
            background: rgba(255, 255, 255, 0.15);
            color: #ffffff;
        }

        .flashcard-tag {
            font-size: 0.7rem;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            color: #fff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load filter options
# -------------------------
filter_options = fetch_filter_options(get_session_id())
tags = normalize_tags(filter_options.get("tags", []))
documents = sorted(filter_options.get("documents", []))

tag_names = [tag["name"] for tag in tags]
tag_color_map = {tag["name"]: tag["color"] for tag in tags}

# -------------------------
# Flash messages
# -------------------------
if st.session_state.flash_success:
    st.success(st.session_state.flash_success)
    st.session_state.flash_success = None

if st.session_state.flash_error:
    st.error(st.session_state.flash_error)
    st.session_state.flash_error = None

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    page_choice = st.radio(
        "View",
        options=["💬 Chat", "🃏 Flashcards", "📊 Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
        key="page_selector_widget",
    )
    if "Flashcards" in page_choice:
        st.session_state.current_page = "flashcards"
    elif "Evaluation" in page_choice:
        st.session_state.current_page = "eval"
    else:
        st.session_state.current_page = "chat"
    st.markdown("---")

    st.markdown('<div class="small-section-title">Upload a PDF</div>', unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )
    st.caption(
        "Uploads are private to this browser session and can be cleared anytime."
    )

    tag_mode = st.radio(
        "Tag mode",
        options=["Use existing tag", "Create new tag"],
        index=0 if tag_names else 1,
    )

    resolved_tag_name = None
    resolved_tag_color = "#2563EB"

    if tag_mode == "Use existing tag":
        if tag_names:
            selected_existing_tag = st.selectbox(
                "Existing tag",
                options=tag_names,
            )
            resolved_tag_name = selected_existing_tag
            resolved_tag_color = tag_color_map.get(selected_existing_tag, "#2563EB")

            st.caption("Selected tag")
            tag_chip(resolved_tag_name, resolved_tag_color)
        else:
            st.info("No existing tags yet. Create a new tag first.")
            tag_mode = "Create new tag"

    if tag_mode == "Create new tag":
        new_tag_name = st.text_input("New tag name")
        resolved_tag_name = new_tag_name.strip() if new_tag_name.strip() else None

        resolved_tag_color = st.color_picker("Tag color", value="#2563EB")

        if resolved_tag_name:
            st.caption("New tag preview")
            tag_chip(resolved_tag_name, resolved_tag_color)

    source_type = st.selectbox(
        "Source type",
        options=["lecture", "textbook", "exam", "assignment", "notes", "other"],
        index=0,
    )

    if st.button("Ingest PDF", use_container_width=True):
        if not uploaded_pdf:
            st.session_state.flash_error = "Please choose a PDF first."
            st.rerun()

        if not resolved_tag_name:
            resolved_tag_name = "Untagged"
            resolved_tag_color = "#64748B"

        try:
            files = {
                "file": (
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue(),
                    "application/pdf",
                )
            }

            data = {
                "tag_name": resolved_tag_name,
                "tag_color": resolved_tag_color,
                "source_type": source_type,
                "session_id": get_session_id(),
            }

            with st.spinner(
                f'Ingesting "{uploaded_pdf.name}" — parsing, embedding, and indexing. '
                "This may take 30–60 seconds for large files."
            ):
                r = requests.post(
                    f"{BACKEND_URL}/ingest_pdf",
                    files=files,
                    data=data,
                    timeout=300,
                )

            if r.ok:
                payload = r.json()
                doc_name = payload.get("doc_name", uploaded_pdf.name)
                chunks = payload.get("chunks", "?")

                st.session_state.flash_success = (
                    f'Ingested "{doc_name}" successfully ({chunks} chunks).'
                )
                fetch_filter_options.clear()

                # Reset uploader box
                st.session_state.uploader_key += 1
                st.rerun()
            else:
                st.session_state.flash_error = f"Ingest failed: {r.status_code} {r.text}"
                st.rerun()

        except Exception as e:
            st.session_state.flash_error = f"Ingest error: {e}"
            st.rerun()

    if st.button("Clear my uploaded documents", use_container_width=True):
        try:
            r = requests.post(
                f"{BACKEND_URL}/clear_session_uploads",
                json={"session_id": get_session_id()},
                timeout=60,
            )

            if r.ok:
                payload = r.json()
                deleted = payload.get("deleted_pdf_files", 0)
                st.session_state.flash_success = (
                    f"Cleared your private uploads ({deleted} PDF files removed)."
                )
                fetch_filter_options.clear()
                st.session_state.doc_filter_select = "All"
                st.session_state.last_hits = []
                st.session_state.last_confidence = None
                st.session_state.flashcards = []
                st.session_state.flashcards_doc = None
                st.rerun()
            else:
                st.session_state.flash_error = (
                    f"Clear failed: {r.status_code} {r.text}"
                )
                st.rerun()
        except Exception as e:
            st.session_state.flash_error = f"Clear error: {e}"
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="small-section-title">System Intelligence</div>', unsafe_allow_html=True)

    if st.session_state.last_confidence:
        confidence_badge(st.session_state.last_confidence)

        with st.expander("Confidence details", expanded=False):
            c = st.session_state.last_confidence

            st.caption(f"Retrieval signal: {format_score(c.get('retrieval_score'), 3)}")
            st.caption(f"Rerank signal: {format_score(c.get('rerank_score'), 3)}")
            st.caption(f"Judge signal: {format_score(c.get('judge_score'), 3)}")

            if c.get("judge_available"):
                st.caption("Judge: available")
            else:
                st.caption("Judge: fallback mode")

            if c.get("judge_reason"):
                st.caption(f"Reason: {c.get('judge_reason')}")

    if st.session_state.last_rewritten_query:
        original = st.session_state.last_original_query or ""
        rewritten = st.session_state.last_rewritten_query
        rewrite_changed = rewritten.strip() != original.strip()

        label = "Rewritten retrieval query" if rewrite_changed else "Retrieval query"
        with st.expander(label, expanded=False):
            if rewrite_changed:
                st.caption("Original")
                st.write(original)
                st.caption("Used for retrieval")
                st.write(rewritten)
            else:
                st.caption("Used for retrieval (rewrite off or unchanged)")
                st.write(rewritten)

    st.markdown("---")
    st.markdown('<div class="small-section-title">Demo documents</div>', unsafe_allow_html=True)

    current_doc = st.session_state.get("doc_filter_select", "All")
    for doc in PRELOADED_DOCS:
        is_active = current_doc == doc["doc_name"]
        label = ("● " if is_active else "○ ") + doc["title"]
        if st.button(label, key=f"sidebar_preloaded_{doc['key']}", use_container_width=True):
            if doc["doc_name"] in documents:
                st.session_state.doc_filter_select = doc["doc_name"]
            else:
                st.session_state.flash_error = f"{doc['title']} is not loaded in Qdrant yet."
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="small-section-title">Citations & Evidence</div>', unsafe_allow_html=True)

    if st.session_state.last_hits:
        for i, h in enumerate(st.session_state.last_hits, start=1):
            doc_name = h.get("doc_name", "Document")
            page_start = h.get("page_start")
            page_end = h.get("page_end")
            tag_name = h.get("tag_name")
            tag_color = h.get("tag_color", "#2563EB")
            source_type = (h.get("source_type") or "").strip().lower()

            if page_start and page_end and page_start != page_end:
                page_label = f"pp. {page_start}–{page_end}"
            elif page_start:
                page_label = f"p. {page_start}"
            else:
                page_label = "page —"

            short_doc = shorten_label(doc_name, 25)

            source_icon = {
                "textbook": "📖",
                "book": "📖",
                "notes": "📝",
                "lecture": "🎓",
                "paper": "📄",
                "article": "📄",
                "slides": "📊",
            }.get(source_type, "📄")

            tag_html = ""
            if tag_name:
                safe_tag = tag_name.replace("<", "&lt;").replace(">", "&gt;")
                tag_html = (
                    f'<span class="citation-header-tag" '
                    f'style="background:{tag_color}33;border:1px solid {tag_color};">'
                    f'{safe_tag}</span>'
                )

            safe_doc_full = doc_name.replace('"', "&quot;")
            safe_doc_short = short_doc.replace("<", "&lt;").replace(">", "&gt;")

            pdf_url_with_page = build_pdf_url(doc_name, page_start)

            st.markdown(
                f"""
                    <div class="citation-header">
                      <span class="citation-header-icon">{source_icon}</span>
                      <a class="citation-header-doc" href="{pdf_url_with_page}" target="_blank" rel="noopener" title="{safe_doc_full} — open PDF at page {page_start or '1'}">{i}. {safe_doc_short}</a>
                      <a class="citation-header-page" href="{pdf_url_with_page}" target="_blank" rel="noopener" title="Open PDF at this page">{page_label}</a>
                      {tag_html}
                    </div>
                    """,
                unsafe_allow_html=True,
            )

            with st.expander("Show passage", expanded=False):
                score_cols = st.columns(2)
                with score_cols[0]:
                    for label, val in (
                            ("Retrieval", h.get("retrieval_score")),
                            ("Dense", h.get("dense_score")),
                            ("Sparse", h.get("sparse_score")),
                    ):
                        st.markdown(
                            f'<div class="citation-score-line citation-score-line-raw">'
                            f'<span class="citation-score-label">{label}</span>'
                            f'<span class="citation-score-val">{format_score(val, 2)}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                with score_cols[1]:
                    st.markdown(
                        f'<div class="citation-score-line citation-score-line-raw">'
                        f'<span class="citation-score-label">Rerank</span>'
                        f'<span class="citation-score-val">{format_score(h.get("rerank_score"), 2)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="citation-score-line citation-score-line-norm">'
                        f'<span class="citation-score-label">Retrieval norm</span>'
                        f'<span class="citation-score-val">{format_score(h.get("retrieval_score_normalized"), 2)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="citation-score-line citation-score-line-norm">'
                        f'<span class="citation-score-label">Rerank norm</span>'
                        f'<span class="citation-score-val">{format_score(h.get("rerank_score_normalized"), 2)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.write(h.get("text", "").strip() or "_No text available_")
    else:
        st.info("Ask a question to see citations and snippets here.")

    st.markdown("---")
    st.toggle(
        "Stream answers",
        key="streaming_enabled",
        help="Tokens appear in real time. Turn off if streaming feels unstable.",
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_hits = []
        st.session_state.last_original_query = None
        st.session_state.last_rewritten_query = None
        st.session_state.last_confidence = None
        st.rerun()


# -------------------------
# Evaluation page
# -------------------------
EVAL_DATA = [
    {
        "title": "Phase 2 — Academic QA benchmark",
        "subtitle": (
            "100 questions across four documents, comparing four retrieval / "
            "generation modes."
        ),
        "date": "Initial baseline",
        "num_questions": 100,
        "judge_model": "gpt-4.1-mini",
        "generator_model": "gpt-4.1-mini",
        "rewrite_model": "gpt-4.1-mini",
        "documents": [
            "Reinforcement Learning: An Introduction (Sutton & Barto)",
            "Operating Systems: Three Easy Pieces",
            "CSC263 — Data Structures and Analysis (course notes)",
            "MAT102 — Introduction to Mathematical Proofs (course notes)",
        ],
        "rows": [
            {"Mode": "Dense Only",                "Faithfulness": 0.956, "Answer Relevancy": 0.886, "Context Precision": 0.780, "Context Recall": 0.958, "Judge Grounding": 0.981, "Judge Correctness": 0.990, "Judge Usefulness": 0.987},
            {"Mode": "Hybrid",                    "Faithfulness": 0.957, "Answer Relevancy": 0.893, "Context Precision": 0.810, "Context Recall": 0.982, "Judge Grounding": 0.990, "Judge Correctness": 1.000, "Judge Usefulness": 0.991},
            {"Mode": "Hybrid + Rerank",           "Faithfulness": 0.962, "Answer Relevancy": 0.894, "Context Precision": 0.830, "Context Recall": 0.982, "Judge Grounding": 0.993, "Judge Correctness": 1.000, "Judge Usefulness": 0.991},
            {"Mode": "Hybrid + Rerank + Rewrite", "Faithfulness": 0.956, "Answer Relevancy": 0.898, "Context Precision": 0.828, "Context Recall": 0.977, "Judge Grounding": 0.993, "Judge Correctness": 1.000, "Judge Usefulness": 0.992},
        ],
        "takeaway": (
            "Hybrid retrieval improves recall over dense-only retrieval; CrossEncoder "
            "reranking improves context precision and answer faithfulness. Query rewriting "
            "nudges answer relevancy higher but trades a hair of faithfulness. "
            "**Best overall configuration: Hybrid + Rerank.** Judge scores cluster near the "
            "top of the scale across all modes, so retrieval-quality metrics (Context "
            "Precision / Recall / Faithfulness) carry more of the discriminative signal."
        ),
    },
]


def render_eval_page() -> None:
    """
    Render the standalone evaluation results page.

    Reads from the EVAL_DATA list so new benchmarks can be appended as
    additional dicts without changing the rendering logic.
    """
    st.markdown("# Evaluation")
    st.markdown(
        "Internal benchmarks comparing retrieval and generation configurations "
        "of Lucid. All numbers are computed on held-out academic-QA "
        "benchmarks using "
        "[RAGAS](https://github.com/explodinggradients/ragas) metrics plus a "
        "custom OpenAI LLM-as-judge for grounding, correctness, and usefulness."
    )
    st.caption(
        "These results describe the system on a finite benchmark. Treat them as "
        "a calibration of relative configuration quality, not an absolute "
        "quality claim that generalizes to every question or document."
    )

    for bench in EVAL_DATA:
        st.markdown("---")
        st.markdown(f"## {bench['title']}")
        st.markdown(bench["subtitle"])

        meta_cols = st.columns(4)
        meta_cols[0].metric("Questions", bench["num_questions"])
        meta_cols[1].metric("Documents", len(bench["documents"]))
        meta_cols[2].metric("Modes", len(bench["rows"]))
        meta_cols[3].metric("Judge", bench["judge_model"])

        with st.expander("Documents & models in this benchmark", expanded=False):
            for doc in bench["documents"]:
                st.markdown(f"- {doc}")
            st.caption(
                f"Generator: `{bench['generator_model']}` • "
                f"Query rewrite: `{bench['rewrite_model']}` • "
                f"Judge: `{bench['judge_model']}` • "
                f"Run: {bench['date']}"
            )

        df = pd.DataFrame(bench["rows"]).set_index("Mode")
        metric_cols = list(df.columns)

        st.markdown("### Comparison")
        styler = (
            df.style
              .format("{:.3f}")
              .highlight_max(
                  subset=metric_cols,
                  axis=0,
                  props=(
                      "background-color: rgba(34,197,94,0.22); "
                      "font-weight: 700; color: #ffffff;"
                  ),
              )
        )
        st.dataframe(styler, use_container_width=True)
        st.caption("Best value per metric highlighted.")

        st.markdown("### RAGAS metrics by mode")
        ragas_metrics = [
            "Faithfulness",
            "Answer Relevancy",
            "Context Precision",
            "Context Recall",
        ]
        st.bar_chart(df[ragas_metrics], use_container_width=True)

        st.markdown("### Takeaway")
        st.markdown(bench["takeaway"])

    st.markdown("---")
    st.caption("More benchmarks will be added here as they're run.")


# -------------------------
# Flashcards page
# -------------------------
def fetch_flashcards(doc_name: str, tag_name: str | None, num_cards: int) -> list[dict]:
    """
    Request flashcards for a document from the backend.

    Args:
        doc_name: Document name to generate cards from.
        tag_name: Optional tag filter.
        num_cards: Requested number of cards.

    Returns:
        Generated flashcard dictionaries.

    Raises:
        RuntimeError: If the backend returns a non-2xx response.
    """
    payload = {
        "doc_name": doc_name,
        "tag_name": None if (not tag_name or tag_name == "All") else tag_name,
        "num_cards": num_cards,
        "session_id": get_session_id(),
    }
    r = requests.post(
        f"{BACKEND_URL}/generate_flashcards",
        json=payload,
        timeout=180,
    )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json().get("cards", []) or []


def render_flashcards_page() -> None:
    """
    Render the flashcard generation and review page.
    """
    st.markdown("# Flashcards")
    st.markdown(
        "Generate study flashcards from a selected document. Each card shows a "
        "question on the front; click to flip and see the answer plus its citation."
    )

    if not documents:
        st.info(
            "No documents indexed yet. Upload a PDF or load a demo document first."
        )
        return

    # Default to the doc currently selected in chat, else the first available
    default_doc = st.session_state.get("doc_filter_select")
    if default_doc == "All" or default_doc not in documents:
        default_doc = documents[0]

    ctrl_cols = st.columns([3, 2, 1.5, 1.5])
    with ctrl_cols[0]:
        chosen_doc = st.selectbox(
            "Document",
            options=documents,
            index=documents.index(default_doc),
            format_func=lambda x: shorten_label(x, 32),
            key="flashcards_doc_select",
        )
    with ctrl_cols[1]:
        chosen_tag = st.selectbox(
            "Tag (optional)",
            options=["All"] + tag_names,
            index=0,
            format_func=lambda x: shorten_label(x, 18),
            key="flashcards_tag_select",
        )
    with ctrl_cols[2]:
        chosen_count = st.slider(
            "Cards",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            key="flashcards_count",
        )
    with ctrl_cols[3]:
        st.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
        generate_clicked = st.button(
            "Generate",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.flashcards_loading,
        )

    if generate_clicked:
        st.session_state.flashcards_loading = True
        st.session_state.flashcards_error = None
        with st.spinner(f'Generating {chosen_count} flashcards from "{chosen_doc}"...'):
            try:
                cards = fetch_flashcards(chosen_doc, chosen_tag, chosen_count)
                st.session_state.flashcards = cards
                st.session_state.flashcards_doc = chosen_doc
                if not cards:
                    st.session_state.flashcards_error = (
                        "No flashcards were generated. Try a different document or tag."
                    )
            except Exception as e:
                st.session_state.flashcards = []
                st.session_state.flashcards_error = f"Flashcard generation failed: {e}"
        st.session_state.flashcards_loading = False
        st.rerun()

    if st.session_state.flashcards_error:
        st.error(st.session_state.flashcards_error)

    cards = st.session_state.flashcards
    if not cards:
        st.info("Choose a document, then click **Generate** to create flashcards.")
        return

    st.caption(
        f"{len(cards)} cards from **{st.session_state.flashcards_doc}** · "
        "click any card to flip it."
    )

    # Build one big HTML block for the flippable grid (CSS-only flip)
    card_html_parts = []
    for i, c in enumerate(cards):
        front = (c.get("front") or "").replace("<", "&lt;").replace(">", "&gt;")
        back = (c.get("back") or "").replace("<", "&lt;").replace(">", "&gt;")
        doc_name = c.get("doc_name") or ""
        page = c.get("page")
        tag_name = c.get("tag_name")
        tag_color = c.get("tag_color") or "#2563EB"

        safe_doc_short = (
            doc_name if len(doc_name) <= 28 else doc_name[:25] + "..."
        ).replace("<", "&lt;").replace(">", "&gt;")
        page_label = f"p. {page}" if page else "page —"

        pdf_url = build_pdf_url(doc_name, page)

        tag_html = ""
        if tag_name:
            safe_tag = tag_name.replace("<", "&lt;").replace(">", "&gt;")
            tag_html = (
                f'<span class="flashcard-tag" '
                f'style="background:{tag_color}33;border:1px solid {tag_color};">'
                f'{safe_tag}</span>'
            )

        card_html_parts.append(
            f'<label class="flashcard">'
            f'<input type="checkbox" class="flashcard-toggle" />'
            f'<div class="flashcard-inner">'
            f'<div class="flashcard-face flashcard-front">'
            f'<div class="flashcard-index">Card {i + 1}</div>'
            f'<div class="flashcard-front-content">{front}</div>'
            f'<div class="flashcard-hint">Click to reveal answer</div>'
            f'</div>'
            f'<div class="flashcard-face flashcard-back">'
            f'<div class="flashcard-back-content">{back}</div>'
            f'<div class="flashcard-citation-row">'
            f'<a class="flashcard-citation" href="{pdf_url}" target="_blank" rel="noopener" onclick="event.stopPropagation();">'
            f'📄 {safe_doc_short} · {page_label}'
            f'</a>'
            f'{tag_html}'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</label>'
        )

    st.markdown(
        f'<div class="flashcard-grid">{"".join(card_html_parts)}</div>',
        unsafe_allow_html=True,
    )


# -------------------------
# State-aware assistant content rendering
# -------------------------
NO_RESULTS_STRINGS = {
    "I couldn't find relevant information in the provided document.",
    "I don't know based on the provided document.",
}

SOFT_GUARD_STRINGS = {
    "Please enter a question.",
    "Please ask a shorter question.",
}

API_ERROR_STRINGS = {
    "The AI service is not configured correctly.",
    "The app is temporarily rate limited or out of quota. Please try again later.",
    "The request could not be processed. Try asking a shorter or clearer question.",
    "There was an issue contacting the AI service. Please try again.",
    "An unexpected error occurred. Please try again.",
}

API_ERROR_PREFIXES = (
    "Request failed:",
    "API error:",
)


def render_assistant_content(content: str) -> None:
    """
    Route an assistant message to the right widget based on what it is.

    - Known "no results" strings render as info callouts.
    - Soft input guards render as warnings.
    - Known and pattern-matched API/transport errors render as errors.
    - Anything else renders as normal prose.
    """
    text = (content or "").strip()
    if not text:
        st.write(content)
        return

    if text in NO_RESULTS_STRINGS:
        st.info(f"🔍 {text}")
        return

    if text in SOFT_GUARD_STRINGS:
        st.warning(text)
        return

    if text in API_ERROR_STRINGS or any(text.startswith(p) for p in API_ERROR_PREFIXES):
        st.error(text)
        return

    st.write(content)


# -------------------------
# Landing screen
# -------------------------
def render_landing_screen() -> None:
    """
    Render the empty-chat onboarding and demo-document selection screen.
    """
    st.markdown(
        """<div style="width: min(100%, 1600px); margin: 1.5rem auto 0 auto; max-height: 66vh; overflow-y: auto; padding: 0 0.45rem 0 0.15rem;">

<div style="margin: 0 0 1.25rem 0; font-size: 1.22rem; line-height: 1.9; opacity: 0.95; text-align: left; width: 100%;">
Lucid is an AI study assistant for asking grounded questions over academic PDFs. It is built to feel like a real product: select a preloaded technical document or upload your own, ask a question, and get an answer backed by citations, retrieved evidence, confidence scoring, and evaluation-aware system design. Unlike a simple chatbot, Lucid shows how the answer was produced — what sources were retrieved, how they were ranked, and how confident the system is in the response.
</div>

<div style="display: flex; flex-direction: column; gap: 1rem; width: 100%;">

<div style="border: 1px solid rgba(255,255,255,0.10); border-radius: 0.85rem; padding: 1.35rem 1.5rem; background: rgba(255,255,255,0.02); width: 100%; box-sizing: border-box;">
<div style="font-size: 1.22rem; font-weight: 800; margin-bottom: 0.9rem;">
How to use it
</div>
<ul style="margin: 0; padding-left: 1.35rem; font-size: 1.12rem; line-height: 1.9; opacity: 0.95;">
<li>Select one of the preloaded documents below, or upload your own PDFs from the sidebar.</li>
<li>Ask a question using the search bar at the bottom of the page.</li>
<li>Optionally select document and tag filters to refine which sources are searched.</li>
<li>Optionally enable Query Rewrite to let AI refine your question for stronger retrieval quality.</li>
<li>Review the answer, citations, confidence score, and retrieved evidence in the sidebar to judge how well-supported the answer is.</li>
<li>Open the retrieval query panel to see how the system rewrote your question for better search.</li>
</ul>
</div>

<div style="border: 1px solid rgba(255,255,255,0.10); border-radius: 0.85rem; padding: 1.35rem 1.5rem; background: rgba(255,255,255,0.02); width: 100%; box-sizing: border-box;">
<div style="font-size: 1.22rem; font-weight: 800; margin-bottom: 0.9rem;">
What it does
</div>
<ul style="margin: 0; padding-left: 1.35rem; font-size: 1.12rem; line-height: 1.9; opacity: 0.95;">
<li>Answers questions using retrieved evidence from uploaded or preloaded academic PDFs.</li>
<li>Supports multi-document Q&amp;A with document and tag filtering.</li>
<li>Shows citations, source chunks, page ranges, retrieval scores, rerank scores, and confidence.</li>
<li>Uses query rewriting to improve retrieval quality while preserving the original user question for answer generation.</li>
<li>Estimates answer quality using retrieval signals, rerank scores, and an LLM grounding judge.</li>
<li>Includes a serious evaluation pipeline comparing retrieval modes across a curated 100-question dataset.</li>
</ul>
</div>

<div style="border: 1px solid rgba(255,255,255,0.10); border-radius: 0.85rem; padding: 1.35rem 1.5rem; background: rgba(255,255,255,0.02); width: 100%; box-sizing: border-box;">
<div style="font-size: 1.22rem; font-weight: 800; margin-bottom: 0.9rem;">
Under the hood
</div>
<ul style="margin: 0; padding-left: 1.35rem; font-size: 1.12rem; line-height: 1.9; opacity: 0.95;">
<li><b>Ingestion:</b> Docling parses PDFs, then a custom paragraph-aware, token-counted chunker preserves readable context and page metadata.</li>
<li><b>Retrieval:</b> SentenceTransformers dense embeddings + FastEmbed BM25 sparse embeddings are stored in Qdrant.</li>
<li><b>Hybrid search:</b> Dense and sparse results are fused with Reciprocal Rank Fusion for stronger semantic + keyword retrieval.</li>
<li><b>Reranking:</b> A CrossEncoder reranks retrieved chunks before answer generation.</li>
<li><b>Generation:</b> OpenAI gpt-4.1-mini generates citation-aware answers grounded in retrieved sources.</li>
<li><b>System intelligence:</b> query rewriting, grounding judge, confidence scoring, score display, and fail-safe fallbacks.</li>
<li><b>Evaluation:</b> RAGAS metrics + custom LLM-as-judge compare Dense, Hybrid, Hybrid + Rerank, and Hybrid + Rerank + Rewrite modes.</li>
<li><b>Observability:</b> LangSmith traces rewrite, retrieval, reranking, prompt build, generation, judge, confidence, and latency.</li>
<li><b>Stack:</b> FastAPI · Streamlit · Qdrant · OpenAI · SentenceTransformers · FastEmbed · CrossEncoder · Docling · RAGAS · LangSmith.</li>
</ul>
</div>

</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Cards label
    st.markdown(
        '<div style="font-size: 0.85rem; opacity: 0.7; margin: 1.5rem 0 0.6rem 0; '
        'text-transform: uppercase; letter-spacing: 0.08em;">Try a preloaded document</div>',
        unsafe_allow_html=True,
    )

    # Cards loop
    cols = st.columns(4, gap="small")
    for col, doc in zip(cols, PRELOADED_DOCS):
        with col:
            st.markdown(
                f"""
                    <div style="
                        border: 1px solid rgba(255,255,255,0.10);
                        border-left: 3px solid {doc['accent']};
                        border-radius: 0.6rem;
                        padding: 0.85rem 0.9rem;
                        background: rgba(255,255,255,0.02);
                        height: 110px;
                    ">
                        <div style="font-size: 1rem; font-weight: 700; margin-bottom: 0.25rem;">
                            {doc['title']}
                        </div>
                        <div style="font-size: 0.8rem; opacity: 0.7; line-height: 1.35;">
                            {doc['subtitle']}
                        </div>
                    </div>
                    """,
                unsafe_allow_html=True,
            )
            if st.button("Select", key=f"landing_card_{doc['key']}", use_container_width=True):
                if doc["doc_name"] in documents:
                    st.session_state.doc_filter_select = doc["doc_name"]
                    st.session_state.flash_success = (
                        f"Filtered to {doc['title']}. Ask a question below."
                    )
                else:
                    st.session_state.flash_error = (
                        f"{doc['title']} is not loaded in Qdrant yet."
                    )
                st.rerun()

    active_doc = get_preloaded_doc_by_name(st.session_state.get("doc_filter_select"))
    if active_doc:
        render_example_questions(active_doc, location="landing")

    # CTA banner — separate, top-level call
    st.markdown(
        """
        <div style="
            margin: 1.75rem 0 0.5rem 0;
            padding: 0.9rem 1rem;
            border-radius: 0.6rem;
            background: rgba(99, 102, 241, 0.08);
            border: 1px dashed rgba(99, 102, 241, 0.35);
            font-size: 0.95rem;
        ">
            👇 <b>Try a question</b> using the search bar at the bottom, or upload your own PDF from the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Flashcards page gate
# -------------------------
if st.session_state.current_page == "flashcards":
    render_flashcards_page()
    st.stop()


# -------------------------
# Eval page gate
# -------------------------
if st.session_state.current_page == "eval":
    render_eval_page()
    st.stop()


# -------------------------
# Landing screen OR chat history
# -------------------------
if not documents:
    st.info(
        "📄 No documents indexed yet. Upload a PDF using the sidebar, or run the "
        "preload script to load the demo documents, before asking a question."
    )

show_landing = (
        len(st.session_state.messages) == 0
        and st.session_state.pending_question is None
)

if show_landing:
    render_landing_screen()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_content(msg["content"])
        else:
            st.write(msg["content"])

        confidence = msg.get("confidence")
        if msg["role"] == "assistant" and confidence:
            st.caption(
                f"Confidence: {confidence.get('label', 'Unknown')} "
                f"({format_score(confidence.get('score'), 2)})"
            )

        rewritten_query = msg.get("rewritten_query")
        original_query = msg.get("original_query")

        if msg["role"] == "assistant" and rewritten_query:
            rewrite_changed = (rewritten_query or "").strip() != (original_query or "").strip()
            with st.expander("Retrieval query", expanded=False):
                if rewrite_changed:
                    st.caption("Original")
                    st.write(original_query or "")
                    st.caption("Used for retrieval")
                    st.write(rewritten_query)
                else:
                    st.caption("Used for retrieval (rewrite off or unchanged)")
                    st.write(rewritten_query)

        citations = msg.get("citations", [])
        if msg["role"] == "assistant" and citations:
            st.caption("Citations: " + "  ".join(citations))

        trace_url = msg.get("trace_url")
        if msg["role"] == "assistant" and trace_url:
            st.markdown(
                f'<a href="{trace_url}" target="_blank" rel="noopener" '
                f'style="font-size:0.78rem; opacity:0.7; text-decoration:none;">'
                f'🔗 View LangSmith trace</a>',
                unsafe_allow_html=True,
            )

if not show_landing:
    active_doc_chat = get_preloaded_doc_by_name(st.session_state.get("doc_filter_select"))
    if active_doc_chat:
        render_example_questions(active_doc_chat, location="chat")

# -------------------------
# Bottom search bar + filters
# -------------------------
def _consume_stream(response, capture: dict):
    """
    Yield token strings from an NDJSON streaming response. Populates `capture`
    in place with metadata, the final answer, and confidence so the caller
    can persist them to session_state after st.write_stream completes.
    """
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")

        if event_type == "metadata":
            capture["citations"] = event.get("citations", []) or []
            capture["hits"] = event.get("hits", []) or []
            capture["original_query"] = event.get("original_query")
            capture["rewritten_query"] = event.get("rewritten_query")
            capture["rewrite_used"] = event.get("rewrite_used", False)
            capture["rewrite_error"] = event.get("rewrite_error")
            capture["trace_url"] = event.get("trace_url")

        elif event_type == "token":
            # Yield the whole token delta as it arrives from the backend.
            # The backend streams real OpenAI deltas — adding a per-character
            # sleep here on top of that would compound to several seconds of
            # artificial typing delay on a long answer.
            text = event.get("text", "")
            if text:
                yield text

        elif event_type == "error":
            msg = event.get("message", "An error occurred.")
            capture["error"] = msg
            yield f"\n\n_{msg}_"

        elif event_type == "done":
            capture["answer"] = event.get("answer", "")
            capture["confidence"] = event.get("confidence")


def submit_question_streaming(
        question: str,
        selected_tag_filter: str,
        selected_doc_filter: str,
        rewrite_enabled: bool,
) -> None:
    """
    Stream an answer into a new assistant chat bubble.

    On any pre-stream failure (network, non-2xx) this falls back to the
    non-streaming submit_question for the same request so the user always
    gets an answer or a clear error.

    Args:
        question: User question.
        selected_tag_filter: Selected tag filter, or "All".
        selected_doc_filter: Selected document filter, or "All".
        rewrite_enabled: Whether query rewriting is enabled.
    """
    question = question.strip()
    if not question:
        return

    payload = {
        "question": question,
        "tag_name": None if selected_tag_filter == "All" else selected_tag_filter,
        "doc_name": None if selected_doc_filter == "All" else selected_doc_filter,
        "history": build_history_payload(st.session_state.messages),
        "rewrite_enabled": rewrite_enabled,
        "session_id": get_session_id(),
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/ask_question_stream",
            json=payload,
            stream=True,
            timeout=120,
        )

        if not response.ok:
            raise RuntimeError(f"API error: {response.status_code}")

        capture: dict = {}

        with st.chat_message("assistant"):
            streamed_text = st.write_stream(_consume_stream(response, capture))

        final_answer = capture.get("answer") or streamed_text or ""
        citations = capture.get("citations", []) or []
        hits = capture.get("hits", []) or []

        if "I don't know based on the provided document" in final_answer:
            citations = []
            hits = []

        st.session_state.last_hits = hits
        st.session_state.last_original_query = capture.get("original_query")
        st.session_state.last_rewritten_query = capture.get("rewritten_query")
        st.session_state.last_confidence = capture.get("confidence")

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "citations": citations,
            "hits": hits,
            "original_query": capture.get("original_query"),
            "rewritten_query": capture.get("rewritten_query"),
            "confidence": capture.get("confidence"),
            "trace_url": capture.get("trace_url"),
        })

    except Exception:
        submit_question(
            question=question,
            selected_tag_filter=selected_tag_filter,
            selected_doc_filter=selected_doc_filter,
            rewrite_enabled=rewrite_enabled,
            append_user=False,
        )


def submit_question(
        question: str,
        selected_tag_filter: str,
        selected_doc_filter: str,
        rewrite_enabled: bool,
        append_user: bool = True,
) -> None:
    """
    Submit a question through the non-streaming backend endpoint.

    Args:
        question: User question.
        selected_tag_filter: Selected tag filter, or "All".
        selected_doc_filter: Selected document filter, or "All".
        rewrite_enabled: Whether query rewriting is enabled.
        append_user: Whether to append the user message before submitting.
    """
    question = question.strip()

    if not question:
        return

    if append_user:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
                "citations": [],
                "hits": [],
            }
        )

    try:
        payload = {
            "question": question,
            "tag_name": None if selected_tag_filter == "All" else selected_tag_filter,
            "doc_name": None if selected_doc_filter == "All" else selected_doc_filter,
            "history": build_history_payload(st.session_state.messages),
            "rewrite_enabled": rewrite_enabled,
            "session_id": get_session_id(),
        }

        r = requests.post(
            f"{BACKEND_URL}/ask_question",
            json=payload,
            timeout=120,
        )

        if not r.ok:
            raise RuntimeError(f"API error: {r.status_code} {r.text}")

        data = r.json()

        answer = data.get("answer", "")
        citations = data.get("citations", []) or []
        hits = data.get("hits", []) or []
        original_query = data.get("original_query")
        rewritten_query = data.get("rewritten_query")
        confidence = data.get("confidence")

        if "I don't know based on the provided document" in answer:
            citations = []
            hits = []

        st.session_state.last_hits = hits
        st.session_state.last_original_query = original_query
        st.session_state.last_rewritten_query = rewritten_query
        st.session_state.last_confidence = confidence

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "hits": hits,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "confidence": confidence,
                "trace_url": data.get("trace_url"),
            }
        )

    except Exception as e:
        err = f"Request failed: {e}"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": err,
                "citations": [],
                "hits": [],
                "original_query": None,
                "rewritten_query": None,
                "confidence": None,
            }
        )


st.markdown(
    """
    <style>
        .block-container {
            padding-bottom: 8rem;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker) {
            position: fixed;
            bottom: 1rem;
            left: calc(var(--sb-sidebar-width, 20rem) + 1rem);
            right: 1rem;
            z-index: 999;

            display: flex !important;
            align-items: center !important;
            gap: 0.35rem !important;

            background: #262730;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 0.75rem;
            padding: 0.5rem 0.55rem;

            box-sizing: border-box;
            max-width: calc(100vw - var(--sb-sidebar-width, 20rem) - 2rem);
            overflow: visible;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"]:nth-of-type(2),
        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"]:nth-of-type(3),
        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"]:nth-of-type(4) {
            border-left: 1px solid rgba(255, 255, 255, 0.14);
            padding-left: 0.55rem !important;
        }

        /* Taller search input */
        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        .stTextInput input {
            height: 2.75rem !important;
            font-size: 0.98rem !important;
            padding: 0.4rem 0.6rem !important;
        }

        /* Vertical rewrite toggle: label on top, switch below */
        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"]:nth-of-type(1) {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.15rem !important;
            padding: 0 !important;
        }

        .rewrite-toggle-label {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.68rem;
            font-weight: 600;
            opacity: 0.75;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            line-height: 1;
            margin: 0;
            white-space: nowrap;
        }

        .rewrite-help {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 0.85rem;
            height: 0.85rem;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.12);
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.6rem;
            font-weight: 700;
            cursor: help;
            text-transform: none;
            letter-spacing: 0;
        }

        .rewrite-help:hover {
            background: rgba(255, 255, 255, 0.22);
        }

        .rewrite-help::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: calc(100% + 0.6rem);
            left: -0.5rem;
            transform: none;
            width: 260px;
            padding: 0.55rem 0.7rem;
            background: #1f2028;
            color: #f1f1f1;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 0.4rem;
            font-size: 0.75rem;
            font-weight: 400;
            line-height: 1.4;
            text-transform: none;
            letter-spacing: 0;
            white-space: normal;
            text-align: left;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.45);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease;
            z-index: 999999;
        }

        .rewrite-help:hover::after {
            opacity: 1;
        }

        /* Center the toggle switch under the label */
        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stToggle"] {
            display: flex !important;
            justify-content: center !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"]:nth-of-type(5) {
            display: none !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker) > div {
            min-width: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-testid="stColumn"] {
            min-width: 0 !important;
            padding-right: 0 !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-baseweb="input"] {
            background: transparent !important;
            border: none !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        div[data-baseweb="select"] {
            min-width: 0 !important;
            width: 100% !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        [data-baseweb="select"] > div {
            min-width: 0 !important;
            overflow: hidden !important;
            white-space: nowrap !important;
            text-overflow: ellipsis !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        .stTextInput,
        div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker)
        .stSelectbox {
            margin-bottom: 0 !important;
        }

        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }

        @media (max-width: 900px) {
            div[data-testid="stHorizontalBlock"]:has(.bottom-search-marker) {
                left: 0.75rem;
                right: 0.75rem;
                max-width: calc(100vw - 1.5rem);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.is_thinking and st.session_state.thinking_message:
    st.info(st.session_state.thinking_message)

is_processing = st.session_state.pending_question is not None

with st.form("question_form", clear_on_submit=True):
    col_rewrite, col_question, col_tag, col_doc, col_submit = st.columns(
        [0.55, 6.7, 1.2, 1.8, 0.01],
        gap="small",
    )

    with col_rewrite:
        st.markdown('<span class="bottom-search-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rewrite-toggle-label">'
            'Rewrite'
            '<span class="rewrite-help" data-tooltip="When ON, your question is rewritten into a stronger retrieval query before search. The original question is still used for answer generation.">?</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        rewrite_on = st.toggle(
            "Rewrite",
            key="rewrite_enabled",
            help="When ON, your question is rewritten into a stronger retrieval query before search. The original question is still used for answer generation.",
            disabled=is_processing,
            label_visibility="collapsed",
        )

    with col_question:
        question = st.text_input(
            "Question",
            placeholder="Ask a question about your PDF(s)",
            label_visibility="collapsed",
            disabled=is_processing,
        )

    with col_tag:
        selected_tag_filter = st.selectbox(
            "Tag",
            options=["All"] + tag_names,
            index=0,
            label_visibility="collapsed",
            format_func=lambda x: shorten_label(x, 14),
            disabled=is_processing,
        )

    with col_doc:
        doc_options = ["All"] + documents

        if st.session_state.get("doc_filter_select") not in doc_options:
            st.session_state.doc_filter_select = "All"

        selected_doc_filter = st.selectbox(
            "Document",
            options=doc_options,
            key="doc_filter_select",
            label_visibility="collapsed",
            format_func=lambda x: shorten_label(x, 20),
            disabled=is_processing,
        )

    with col_submit:
        submitted = st.form_submit_button("Send")

if submitted and question.strip() and not is_processing:
    clean_question = question.strip()

    if not documents:
        st.session_state.flash_error = (
            "No documents indexed yet. Upload a PDF or load a demo document before asking."
        )
        st.rerun()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_question,
            "citations": [],
            "hits": [],
        }
    )

    st.session_state.pending_question = clean_question
    st.session_state.pending_tag_filter = selected_tag_filter
    st.session_state.pending_doc_filter = selected_doc_filter
    st.session_state.pending_rewrite_enabled = rewrite_on

    st.session_state.is_thinking = True
    st.session_state.thinking_message = random.choice(SPINNER_MESSAGES)

    st.rerun()

if st.session_state.pending_question:
    if st.session_state.streaming_enabled:
        submit_question_streaming(
            question=st.session_state.pending_question,
            selected_tag_filter=st.session_state.pending_tag_filter,
            selected_doc_filter=st.session_state.pending_doc_filter,
            rewrite_enabled=bool(st.session_state.pending_rewrite_enabled),
        )
    else:
        submit_question(
            question=st.session_state.pending_question,
            selected_tag_filter=st.session_state.pending_tag_filter,
            selected_doc_filter=st.session_state.pending_doc_filter,
            rewrite_enabled=bool(st.session_state.pending_rewrite_enabled),
            append_user=False,
        )

    st.session_state.pending_question = None
    st.session_state.pending_tag_filter = None
    st.session_state.pending_doc_filter = None
    st.session_state.pending_rewrite_enabled = None
    st.session_state.is_thinking = False
    st.session_state.thinking_message = None

    st.rerun()
