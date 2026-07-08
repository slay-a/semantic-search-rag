"""Semantic Search RAG chatbot — Streamlit UI.

Features
--------
* **Workspaces** — each workspace is an isolated knowledge base (multi-tenant).
* **Multi-source ingest** — upload PDF / Word-ish text / CSV / Excel, or add a URL.
* **Grounded chat** — streamed, cited answers with conversation memory.
* **Human handoff** — when the bot can't answer from the docs, it offers escalation.

Run it:  streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make `src/rag_pipeline` importable when run as `streamlit run app.py` from the
# repo root without an editable install (pytest handles this via pyproject).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from rag_pipeline.config import settings
from rag_pipeline.ingest import build_store
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.store_factory import open_for_read, open_for_write

st.set_page_config(page_title="Semantic Search RAG", layout="wide")

PG = settings.store_backend == "pgvector"

# --------------------------------------------------------------------------- #
# Premium Dark Theme — single theme, no toggle
# --------------------------------------------------------------------------- #
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Globals ─────────────────────────────────────────────────────────── */
:root {
  --bg:       #0a0a0c;
  --surface:  #141418;
  --surface2: #1c1c22;
  --surface3: #26262e;
  --text:     #ececf1;
  --muted:    #8e8ea0;
  --border:   rgba(255,255,255,0.08);
  --accent:   #818cf8;
  --accent2:  #a78bfa;
  --glow:     rgba(129, 140, 248, 0.25);
}
*, *::before, *::after { box-sizing: border-box; }
.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
h1,h2,h3,h4,p,span,label,li { color: var(--text); font-family: 'Inter', sans-serif; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: var(--text);
  font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] .stDivider {
  border-color: var(--border);
}

/* ── Brand block ─────────────────────────────────────────────────────── */
.nx-brand {
  display: flex; align-items: center; gap: 0.85rem;
  padding: 0.25rem 0 1.25rem;
}
.nx-logo {
  width: 46px; height: 46px; border-radius: 14px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; color: #fff; font-weight: 800;
  box-shadow: 0 0 24px var(--glow);
}
.nx-title {
  font-size: 1.4rem; font-weight: 800; letter-spacing: -0.03em;
  background: linear-gradient(135deg, var(--text) 40%, var(--accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nx-sub { color: var(--muted); font-size: 0.78rem; font-weight: 500; letter-spacing: 0.02em; }

/* ── Section labels (sidebar) ────────────────────────────────────────── */
.nx-section {
  display: flex; align-items: center; gap: 0.45rem;
  font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted);
  margin: 1.25rem 0 0.5rem; padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
}

/* ── Expanders ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  background: var(--surface2); padding: 0.6rem 0.9rem; font-weight: 600;
}
[data-testid="stExpander"] summary p { font-weight: 600; font-size: 0.9rem; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton>button {
  border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface2); color: var(--text);
  font-weight: 600; font-family: 'Inter', sans-serif;
  transition: all 0.2s ease; padding: 0.45rem 1rem;
}
.stButton>button:hover {
  border-color: var(--accent); background: var(--surface3);
  box-shadow: 0 0 12px var(--glow);
}

/* ── Build KB button — special accent ────────────────────────────────── */
.build-btn .stButton>button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  border: none !important; color: #fff !important;
}
.build-btn .stButton>button:hover { filter: brightness(1.12); }

/* ── Chat messages ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 0.85rem;
}
[data-testid="stChatMessageContent"] {
  color: var(--text); line-height: 1.7; font-size: 0.95rem;
}
/* User bubble — slightly different tint */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--surface2);
  border-color: rgba(129,140,248,0.2);
}

/* ── Source citation pills ───────────────────────────────────────────── */
.nx-src {
  display: inline-flex; align-items: center;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.2rem 0.55rem;
  margin: 0.2rem 0.3rem 0 0;
  font-size: 0.72rem; color: var(--muted); font-weight: 500;
  transition: all 0.2s ease; cursor: default;
}
.nx-src:hover { border-color: var(--accent); color: var(--text); }

/* ── Human-handoff box ───────────────────────────────────────────────── */
.nx-handoff {
  background: var(--surface2);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 0.9rem 1.1rem; margin-top: 0.65rem;
  font-size: 0.9rem; color: var(--text);
}

/* ── Chat input ──────────────────────────────────────────────────────── */
div[data-testid="stChatInput"] { padding-bottom: 1.5rem; }
div[data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 0.75rem 1rem;
  font-family: 'Inter', sans-serif;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--glow) !important;
}

/* ── Status badges ───────────────────────────────────────────────────── */
.nx-badge {
  display: inline-block; font-size: 0.62rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); background: var(--surface2);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 0.18rem 0.45rem; margin-right: 0.35rem;
}

/* ── Main header area ────────────────────────────────────────────────── */
.nx-header {
  padding: 0.5rem 0 1.5rem;
}
.nx-header h2 {
  font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em;
  margin: 0 0 0.2rem;
}
.nx-header-sub {
  font-size: 0.85rem; color: var(--muted); font-weight: 500;
}
.nx-header-sub strong { color: var(--accent); font-weight: 600; }

/* ── Empty-state card ────────────────────────────────────────────────── */
.nx-empty {
  text-align: center; padding: 4rem 2rem;
  color: var(--muted);
}
.nx-empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.nx-empty h3 { color: var(--text); font-weight: 700; margin-bottom: 0.5rem; }
.nx-empty p  { max-width: 400px; margin: 0 auto; line-height: 1.6; font-size: 0.9rem; }

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Popover & Destructive Buttons ────────────────────────────────────── */
div[data-testid="stPopover"] {
  width: 100% !important;
}
div[data-testid="stPopover"] > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  padding: 0.55rem 0.85rem !important;
  font-weight: 500 !important;
  text-align: left !important;
  justify-content: space-between !important;
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  font-family: 'Inter', sans-serif !important;
}
div[data-testid="stPopover"] > button:hover {
  border-color: var(--accent) !important;
  background: var(--surface3) !important;
  box-shadow: 0 0 12px var(--glow) !important;
}
button[data-testid="baseButton-primary"] {
  background: rgba(239, 68, 68, 0.15) !important;
  border: 1px solid rgba(239, 68, 68, 0.3) !important;
  color: #ef4444 !important;
}
button[data-testid="baseButton-primary"]:hover {
  background: #ef4444 !important;
  color: #ffffff !important;
  border-color: #ef4444 !important;
  box-shadow: 0 0 12px rgba(239, 68, 68, 0.3) !important;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


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


def get_workspaces() -> list[str]:
    workspaces = {"default"}
    try:
        if PG:
            store = open_for_read(settings)
            if hasattr(store, "_conn"):
                with store._conn.cursor() as cur:
                    cur.execute("SELECT to_regclass(%s)", (f"public.{store._table}",))
                    if cur.fetchone()[0] is not None:
                        cur.execute(f"SELECT DISTINCT tenant_id FROM {store._table}")
                        rows = cur.fetchall()
                        for r in rows:
                            if r[0]:
                                workspaces.add(r[0])
        else:
            if Path(settings.store_path).exists():
                store = open_for_read(settings)
                for c in store._chunks:
                    tid = c.metadata.get("tenant_id")
                    if tid:
                        workspaces.add(tid)
    except Exception:
        pass
    return sorted(list(workspaces))


def rename_workspace(old_name: str, new_name: str) -> None:
    if not old_name or not new_name or old_name == new_name:
        return
    try:
        if PG:
            store = open_for_write(settings, append=True)
            if hasattr(store, "_conn"):
                with store._conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {store._table} SET tenant_id = %s WHERE tenant_id = %s",
                        (new_name, old_name)
                    )
        else:
            if Path(settings.store_path).exists():
                store = open_for_write(settings, append=True)
                for c in store._chunks:
                    if c.metadata.get("tenant_id") == old_name:
                        c.metadata["tenant_id"] = new_name
                store.save(settings.store_path)
    except Exception:
        pass


def delete_workspace(workspace_name: str) -> None:
    if not workspace_name:
        return
    try:
        if PG:
            store = open_for_write(settings, append=True)
            if hasattr(store, "_conn"):
                with store._conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {store._table} WHERE tenant_id = %s",
                        (workspace_name,)
                    )
        else:
            if Path(settings.store_path).exists():
                store = open_for_write(settings, append=True)
                keep_indices = []
                new_chunks = []
                for i, c in enumerate(store._chunks):
                    if c.metadata.get("tenant_id") != workspace_name:
                        keep_indices.append(i)
                        new_chunks.append(c)
                
                if keep_indices and store._vectors is not None:
                    store._chunks = new_chunks
                    store._vectors = store._vectors[keep_indices]
                else:
                    store._chunks = []
                    store._vectors = None
                
                store.save(settings.store_path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?, handoff?}]
if "workspace" not in st.session_state:
    st.session_state.workspace = "default"
if "workspaces" not in st.session_state:
    st.session_state.workspaces = get_workspaces()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    # ── Brand
    st.markdown(
        f'<div class="nx-brand"><div class="nx-logo">S</div>'
        f'<div><div class="nx-title">{settings.bot_name}</div>'
        f'<div class="nx-sub">Semantic Search · Grounded Answers</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── Workspace selector
    st.markdown('<div class="nx-section">Workspace</div>', unsafe_allow_html=True)
    
    workspaces = list(st.session_state.workspaces)
    if st.session_state.workspace not in workspaces:
        workspaces.append(st.session_state.workspace)
        workspaces = sorted(list(set(workspaces)))
        st.session_state.workspaces = workspaces

    with st.popover(f"{st.session_state.workspace} ▾", use_container_width=True):
        st.markdown("**Switch Workspace**")
        
        # 1. Selection List
        for ws in workspaces:
            is_active = (ws == st.session_state.workspace)
            btn_label = f"• {ws}" if is_active else ws
            if st.button(btn_label, key=f"ws_sel_{ws}", use_container_width=True):
                st.session_state.workspace = ws
                st.rerun()
                
        st.markdown("---")
        
        # 2. Operations Tabs
        tab_new, tab_rename, tab_delete = st.tabs(["New", "Rename", "Delete"])
        
        with tab_new:
            with st.form("add_ws_form", clear_on_submit=True):
                new_ws = st.text_input("Workspace Name", placeholder="e.g. client-abc")
                if st.form_submit_button("Create Workspace", use_container_width=True):
                    new_ws = new_ws.strip().lower().replace(" ", "-")
                    if new_ws:
                        st.session_state.workspace = new_ws
                        st.session_state.workspaces = get_workspaces()
                        st.rerun()
                        
        with tab_rename:
            st.markdown(f"Rename current: `{st.session_state.workspace}`")
            new_name = st.text_input("New Name", value=st.session_state.workspace, key="popover_rename")
            if st.button("Save", key="popover_rename_save", use_container_width=True):
                new_name = new_name.strip().lower().replace(" ", "-")
                if new_name and new_name != st.session_state.workspace:
                    old_name = st.session_state.workspace
                    rename_workspace(old_name, new_name)
                    st.session_state.workspace = new_name
                    load_pipeline.clear()
                    st.session_state.workspaces = get_workspaces()
                    st.rerun()
                    
        with tab_delete:
            st.markdown(f"Delete current: `{st.session_state.workspace}`")
            if st.session_state.workspace == "default":
                st.info("The default workspace cannot be deleted.")
            else:
                st.warning("All data in this workspace will be permanently deleted.")
                if st.button("Delete Workspace", key="popover_delete", use_container_width=True, type="primary"):
                    old_name = st.session_state.workspace
                    delete_workspace(old_name)
                    st.session_state.workspace = "default"
                    load_pipeline.clear()
                    st.session_state.workspaces = get_workspaces()
                    st.rerun()

    tenant_id = st.session_state.workspace

    # ── Persona
    st.markdown('<div class="nx-section">Persona</div>', unsafe_allow_html=True)
    persona = st.text_area(
        "Bot personality (optional)", value=settings.bot_persona,
        placeholder="e.g. You are Ava, a warm, concise support agent for Acme.",
        height=68, label_visibility="collapsed",
    )

    # ── Knowledge Base (in sidebar dropdown)
    st.markdown('<div class="nx-section">Knowledge Base</div>', unsafe_allow_html=True)
    with st.expander("Add sources", expanded=not index_ready(settings.store_path)):
        uploads = st.file_uploader(
            "Upload docs (.pdf .txt .md .csv .xlsx)",
            type=["pdf", "txt", "md", "markdown", "rst", "csv", "xlsx"],
            accept_multiple_files=True,
        )
        url = st.text_input("…or paste a URL", placeholder="https://example.com/faq")
        use_samples = st.checkbox("Include sample docs", value=False)
        append = st.checkbox("Append to existing", value=True)

        with st.container():
            st.markdown('<div class="build-btn">', unsafe_allow_html=True)
            build_clicked = st.button("Build knowledge base", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if build_clicked:
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
                    st.session_state.workspaces = get_workspaces()
                    st.success(f"Knowledge base updated — {len(store)} chunks indexed.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    # ── Retrieval settings
    st.markdown('<div class="nx-section">Retrieval</div>', unsafe_allow_html=True)
    with st.expander("Settings", expanded=False):
        top_k = st.slider("Sources per answer", 1, 15, settings.top_k)
        use_hyde = st.toggle("HyDE query rewriting", value=False)
        st.markdown(
            f'<span class="nx-badge">{settings.embedding_provider}</span>'
            f'<span class="nx-badge">{settings.generation_model}</span>'
            f'<span class="nx-badge">{"hybrid+rerank" if settings.hybrid else "dense"}</span>'
            f'<span class="nx-badge">{settings.store_backend}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
st.markdown(
    f'<div class="nx-header">'
    f'<h2>{settings.bot_name}</h2>'
    f'<div class="nx-header-sub">Workspace: <strong>{tenant_id}</strong> · '
    f'Ask anything about your uploaded knowledge base</div></div>',
    unsafe_allow_html=True,
)

# Show empty state when no messages yet
if not st.session_state.messages:
    st.markdown(
        '<div class="nx-empty">'
        '<h3>Ready to chat</h3>'
        '<p>Upload documents in the sidebar, build your knowledge base, '
        'then ask a question below.</p></div>',
        unsafe_allow_html=True,
    )

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
                '<div class="nx-handoff">I couldn\'t find this in the knowledge '
                "base. Want me to connect you with a human agent?</div>",
                unsafe_allow_html=True,
            )

prompt = st.chat_input("Ask a question…")

if prompt:
    if not index_ready(settings.store_path):
        st.warning("No knowledge base yet — add sources from the sidebar first.")
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
                '<div class="nx-handoff">I couldn\'t find this in the knowledge '
                "base. Want me to connect you with a human agent?</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": streamed,
         "sources": sources, "handoff": answer.abstained}
    )
