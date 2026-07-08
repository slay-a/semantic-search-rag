"""NextCX-style RAG chatbot — Streamlit UI.

Features
--------
* **Workspaces** — each workspace is an isolated knowledge base (multi-tenant).
* **Multi-source ingest** — upload PDF / Word-ish text / CSV / Excel, or add a URL.
* **Grounded chat** — streamed, cited answers with conversation memory.
* **Human handoff** — when the bot can't answer from the docs, it offers escalation.
* **Light / dark themes** — toggle in the sidebar.

Run it:  streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from string import Template

# Make `src/rag_pipeline` importable when run as `streamlit run app.py` from the
# repo root without an editable install (pytest handles this via pyproject).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from rag_pipeline.config import settings
from rag_pipeline.ingest import build_store
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.store_factory import open_for_read, open_for_write

st.set_page_config(page_title="NextCX RAG", page_icon="💬", layout="wide")

PG = settings.store_backend == "pgvector"

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
THEMES = {
    "Dark": {
        "bg": "#0f1117", "panel": "#171a23", "panel2": "#1e2230",
        "text": "#e8e9f0", "muted": "#9aa0b4", "border": "#2a2f3d",
        "accent": "#7c6cf6", "accent_soft": "#241f3d", "user": "#7c6cf6",
        "user_text": "#ffffff",
    },
    "Light": {
        "bg": "#f6f7fb", "panel": "#ffffff", "panel2": "#f0f2f8",
        "text": "#1a1c2b", "muted": "#6b7280", "border": "#e4e7ef",
        "accent": "#6c5ce7", "accent_soft": "#eee9ff", "user": "#6c5ce7",
        "user_text": "#ffffff",
    },
}

_CSS = Template("""
<style>
.stApp { background: $bg; color: $text; }
section[data-testid="stSidebar"] { background: $panel; border-right: 1px solid $border; }
section[data-testid="stSidebar"] * { color: $text; }
h1, h2, h3, h4, p, span, label, li { color: $text; }
.nx-brand { display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem; }
.nx-logo { width:38px; height:38px; border-radius:11px;
  background:linear-gradient(135deg,$accent,#b06cf6); display:flex; align-items:center;
  justify-content:center; font-size:20px; box-shadow:0 4px 14px $accent_soft; }
.nx-title { font-size:1.35rem; font-weight:700; line-height:1; }
.nx-sub { color:$muted; font-size:.8rem; }
/* Chat bubbles */
[data-testid="stChatMessage"] { background:$panel; border:1px solid $border;
  border-radius:16px; padding:.35rem .9rem; margin-bottom:.4rem; }
[data-testid="stChatMessageContent"] { color:$text; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:$accent_soft; border-color:$accent; }
/* Source pills */
.nx-src { display:inline-block; background:$panel2; border:1px solid $border;
  border-radius:999px; padding:.12rem .6rem; margin:.15rem .25rem 0 0;
  font-size:.74rem; color:$muted; }
.nx-handoff { background:$accent_soft; border:1px solid $accent; border-radius:12px;
  padding:.7rem .9rem; margin-top:.5rem; font-size:.9rem; }
div[data-testid="stChatInput"] textarea { background:$panel !important; color:$text !important; }
.stButton>button { border-radius:10px; border:1px solid $border; background:$accent;
  color:#fff; font-weight:600; }
.stButton>button:hover { filter:brightness(1.08); border-color:$accent; }
[data-testid="stExpander"] { border:1px solid $border; border-radius:12px; background:$panel; }
.nx-badge { display:inline-block; font-size:.7rem; color:$muted; border:1px solid $border;
  border-radius:6px; padding:.05rem .4rem; margin-right:.3rem; }
</style>
""")


def inject_theme(name: str) -> None:
    st.markdown(_CSS.substitute(THEMES[name]), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached pipeline + helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_pipeline(store_path: str, use_hyde: bool, persona: str) -> RAGPipeline:
    return RAGPipeline.from_store(store_path, use_hyde=use_hyde, persona=persona)


def index_ready(store_path: str) -> bool:
    if PG:
        try:
            return len(open_for_read(settings)) > 0
        except Exception:
            return False
    return Path(store_path).exists()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?, handoff?}]
if "workspace" not in st.session_state:
    st.session_state.workspace = "default"

inject_theme(st.session_state.theme)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        f'<div class="nx-brand"><div class="nx-logo">💬</div>'
        f'<div><div class="nx-title">{settings.bot_name}</div>'
        f'<div class="nx-sub">NextCX RAG · grounded answers</div></div></div>',
        unsafe_allow_html=True,
    )

    st.session_state.theme = st.radio(
        "Theme", ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
    )

    st.divider()
    workspace = st.text_input(
        "Workspace", value=st.session_state.workspace,
        help="Isolated knowledge base. Uploads and answers stay within it.",
    )
    st.session_state.workspace = workspace or "default"
    tenant_id = st.session_state.workspace

    persona = st.text_area(
        "Bot personality (optional)", value=settings.bot_persona,
        placeholder="e.g. You are Ava, a warm, concise support agent for Acme.",
        height=70,
    )

    with st.expander("📥 Knowledge base", expanded=not index_ready(settings.store_path)):
        uploads = st.file_uploader(
            "Upload docs (.pdf .txt .md .csv .xlsx)",
            type=["pdf", "txt", "md", "markdown", "rst", "csv", "xlsx"],
            accept_multiple_files=True,
        )
        url = st.text_input("…or add a website URL", placeholder="https://example.com/faq")
        use_samples = st.checkbox("Include sample docs", value=False)
        append = st.checkbox("Append (keep existing)", value=True)

        if st.button("Build knowledge base", use_container_width=True):
            paths, urls = [], []
            tmpdir = Path(tempfile.mkdtemp(prefix="nx_upload_"))
            for f in uploads or []:
                dest = tmpdir / f.name
                dest.write_bytes(f.getbuffer())
                paths.append(str(dest))
            if use_samples:
                paths.append("data/sample_docs")
            if url.strip():
                urls.append(url.strip())

            if not paths and not urls:
                st.error("Upload a file, add a URL, or tick sample docs.")
            else:
                try:
                    with st.spinner("Embedding & indexing…"):
                        store = open_for_write(settings, append=append, tenant_id=tenant_id)
                        build_store(paths, urls=urls, store=store, tenant_id=tenant_id)
                        store.save(settings.store_path)
                    load_pipeline.clear()
                    st.success(f"Knowledge base updated ({len(store)} chunks total).")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    with st.expander("⚙️ Retrieval", expanded=False):
        top_k = st.slider("Sources per answer", 1, 15, settings.top_k)
        use_hyde = st.toggle("HyDE query rewriting", value=False)
        st.markdown(
            f'<span class="nx-badge">{settings.embedding_provider}</span>'
            f'<span class="nx-badge">{settings.generation_model}</span>'
            f'<span class="nx-badge">{"hybrid+rerank" if settings.hybrid else "dense"}</span>'
            f'<span class="nx-badge">{settings.store_backend}</span>',
            unsafe_allow_html=True,
        )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main chat
# --------------------------------------------------------------------------- #
st.markdown(f"### 💬 {settings.bot_name}")
st.caption(f"Workspace: **{tenant_id}** · ask anything about your uploaded knowledge base")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                "".join(f'<span class="nx-src">{s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )
        if msg.get("handoff"):
            st.markdown(
                '<div class="nx-handoff">🙋 I couldn\'t find this in the knowledge '
                "base. Want me to connect you with a human agent?</div>",
                unsafe_allow_html=True,
            )

prompt = st.chat_input("Type your question…")

if prompt:
    if not index_ready(settings.store_path):
        st.warning("No knowledge base yet — build one from the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Conversation memory: last few plain turns (exclude the message just added).
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ][-6:]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            rag = load_pipeline(settings.store_path, use_hyde, persona)
            tokens, answer = rag.stream_query(
                prompt, top_k=top_k, tenant_id=tenant_id, history=history
            )
            streamed = ""
            for delta in tokens:
                streamed += delta
                placeholder.markdown(streamed + "▌")
            placeholder.markdown(streamed)
        except Exception as exc:
            placeholder.error(f"Something went wrong: {exc}")
            st.stop()

        sources = [f"[{c.marker}] {Path(c.source).name}" for c in answer.citations]
        if sources:
            st.markdown(
                "".join(f'<span class="nx-src">{s}</span>' for s in sources),
                unsafe_allow_html=True,
            )
        if answer.abstained:
            st.markdown(
                '<div class="nx-handoff">🙋 I couldn\'t find this in the knowledge '
                "base. Want me to connect you with a human agent?</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": streamed,
         "sources": sources, "handoff": answer.abstained}
    )
