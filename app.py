"""Semantic Search RAG chatbot — Streamlit UI.

Features
--------
* **Auth** — optional Supabase email/password login; each account is isolated.
* **Workspaces** — isolated knowledge bases (multi-tenant), switch/create/rename/delete.
* **Multi-source ingest** — PDF / TXT / MD / CSV / Excel, a single URL, or a full site crawl.
* **Grounded chat** — streamed, cited answers with conversation memory.
* **Human handoff** — when the bot can't answer from the docs, it offers escalation.

Run it:  streamlit run app.py
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

# Make `src/rag_pipeline` importable when run as `streamlit run app.py` from the
# repo root without an editable install (pytest handles this via pyproject).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# On Streamlit Community Cloud, config comes from st.secrets. Mirror it into the
# environment BEFORE importing config (which reads os.getenv at import time), so
# the same code path works locally (.env) and on Cloud (secrets).
try:
    import os as _os
    for _k, _v in dict(st.secrets).items():
        if isinstance(_v, str):
            _os.environ.setdefault(_k, _v)
except Exception:
    pass

from rag_pipeline import auth
from rag_pipeline.config import settings
from rag_pipeline.ingest import build_store
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.store_factory import open_for_read, open_for_write

APP_NAME = "ZenRag"
_ASSETS = Path(__file__).resolve().parent / "assets"


def _find_logo() -> Path | None:
    """First logo file present in assets/ (a real image wins over the fallback)."""
    for name in ("zenraglogo.png", "zenraglogo.jpg", "zenraglogo.jpeg", "zenraglogo.png"):
        p = _ASSETS / name
        if p.exists():
            return p
    return None


LOGO_PATH = _find_logo()
# Raster files make good favicons; SVG favicons are unreliable across browsers.
_FAVICON = str(LOGO_PATH) if (LOGO_PATH and LOGO_PATH.suffix.lower() in (".png", ".jpg", ".jpeg")) else None

st.set_page_config(page_title=APP_NAME, page_icon=_FAVICON, layout="wide")

PG = settings.store_backend == "pgvector"

# Built-in fallback logo (used only if no file exists in assets/).
LOGO_SVG = """
<svg width="44" height="44" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="zg" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#fdba74"/><stop offset="0.55" stop-color="#f97316"/>
      <stop offset="1" stop-color="#ea580c"/>
    </linearGradient>
  </defs>
  <rect width="48" height="48" rx="14" fill="url(#zg)"/>
  <circle cx="24" cy="24" r="11" fill="none" stroke="#fff" stroke-width="3"
          stroke-linecap="round" stroke-dasharray="57 100"
          transform="rotate(-42 24 24)" opacity="0.96"/>
  <circle cx="33.2" cy="14.8" r="2.4" fill="#fff"/>
</svg>
"""


def _logo_html(size: int = 44) -> str:
    """Logo markup: the assets/ image (base64-embedded) or the inline fallback."""
    if LOGO_PATH:
        mime = "image/svg+xml" if LOGO_PATH.suffix.lower() == ".svg" else f"image/{LOGO_PATH.suffix.lower().lstrip('.')}"
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return (
            f'<img src="data:{mime};base64,{b64}" alt="{APP_NAME} logo" '
            f'width="{size}" height="{size}" style="object-fit:contain;display:block"/>'
        )
    return LOGO_SVG


LOGO_HTML = _logo_html()

# --------------------------------------------------------------------------- #
# Premium Dark Theme — single theme, no toggle
# --------------------------------------------------------------------------- #
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg:       #0c0c0e;
  --surface:  #16151a;
  --surface2: #1e1d23;
  --surface3: #2a2830;
  --text:     #f3f1ee;
  --muted:    #9a958d;
  --border:   rgba(255,255,255,0.07);
  --accent:   #f97316;   /* ember orange */
  --accent2:  #fbbf24;   /* warm gold */
  --glow:     rgba(249, 115, 22, 0.28);
}
*, *::before, *::after { box-sizing: border-box; }
.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
h1,h2,h3,h4,p,label,li { color: var(--text); }
/* Never override Streamlit's icon font (breaks the chevron/upload glyphs). */
[data-testid="stIconMaterial"], span[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important;
}
/* Hide the "Press Enter to submit" / "Press Enter to apply" hint under inputs. */
[data-testid="InputInstructions"], [data-testid="stInputInstructions"],
[data-testid="stWidgetInstructions"] { display: none !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] { color: var(--text); }
section[data-testid="stSidebar"] .stDivider { border-color: var(--border); }

/* Brand block */
.nx-brand { display: flex; align-items: center; gap: 0.8rem; padding: 0.25rem 0 1.25rem; }
.nx-brand svg { border-radius: 14px; box-shadow: 0 0 22px var(--glow); flex-shrink: 0; }
.nx-brand img { flex-shrink: 0; filter: drop-shadow(0 0 9px var(--glow)); }
.nx-title {
  font-size: 1.4rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;
  background: linear-gradient(135deg, var(--text) 35%, var(--accent) 130%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nx-sub { color: var(--muted); font-size: 0.78rem; font-weight: 500; letter-spacing: 0.02em; }

/* Section labels (sidebar) */
.nx-section {
  display: flex; align-items: center; gap: 0.45rem;
  font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted);
  margin: 1.25rem 0 0.5rem; padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
}
.nx-account { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem; }

/* Expanders */
[data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 10px;
  background: var(--surface); overflow: hidden;
}
[data-testid="stExpander"] summary { background: var(--surface2); padding: 0.6rem 0.9rem; font-weight: 600; }
[data-testid="stExpander"] summary p { font-weight: 600; font-size: 0.9rem; }

/* Buttons */
.stButton>button {
  border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface2); color: var(--text);
  font-weight: 600; font-family: 'Inter', sans-serif;
  transition: all 0.2s ease; padding: 0.45rem 1rem;
}
.stButton>button:hover { border-color: var(--accent); background: var(--surface3); box-shadow: 0 0 12px var(--glow); }

/* Build KB / primary form button — accent gradient */
.build-btn .stButton>button, div[data-testid="stForm"] .stButton>button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  border: none !important; color: #fff !important;
}
.build-btn .stButton>button:hover { filter: brightness(1.12); }

/* Chat messages */
[data-testid="stChatMessage"] {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 0.85rem;
}
[data-testid="stChatMessageContent"] { color: var(--text); line-height: 1.7; font-size: 0.95rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--surface2); border-color: rgba(249,115,22,0.22);
}

/* Source citation pills */
.nx-src {
  display: inline-flex; align-items: center;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.2rem 0.55rem; margin: 0.2rem 0.3rem 0 0;
  font-size: 0.72rem; color: var(--muted); font-weight: 500;
  transition: all 0.2s ease; cursor: default;
}
.nx-src:hover { border-color: var(--accent); color: var(--text); }

/* Human-handoff box */
.nx-handoff {
  background: var(--surface2); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.65rem;
  font-size: 0.9rem; color: var(--text);
}

/* Chat input */
div[data-testid="stChatInput"] { padding-bottom: 1.5rem; }
div[data-testid="stChatInput"] textarea {
  background: var(--surface) !important; color: var(--text) !important;
  border-radius: 12px; border: 1px solid var(--border);
  padding: 0.75rem 1rem; font-family: 'Inter', sans-serif;
}
div[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--glow) !important;
}

/* Status badges */
.nx-badge {
  display: inline-block; font-size: 0.62rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); background: var(--surface2);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 0.18rem 0.45rem; margin-right: 0.35rem;
}

/* Main header area */
.nx-header { padding: 0.5rem 0 1.5rem; }
.nx-header h2 { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 0.2rem; }
.nx-header-sub { font-size: 0.85rem; color: var(--muted); font-weight: 500; }
.nx-header-sub strong { color: var(--accent); font-weight: 600; }

/* Auth / login screen */
.nx-auth-wrap { max-width: 420px; margin: 3.5rem auto 0; }

/* Empty-state card */
.nx-empty { text-align: center; padding: 4rem 2rem; color: var(--muted); }
.nx-empty h3 { color: var(--text); font-weight: 700; margin-bottom: 0.5rem; }
.nx-empty p  { max-width: 400px; margin: 0 auto; line-height: 1.6; font-size: 0.9rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Destructive buttons */
button[data-testid="baseButton-primary"] {
  background: rgba(239, 68, 68, 0.15) !important;
  border: 1px solid rgba(239, 68, 68, 0.3) !important;
  color: #ef4444 !important;
}
button[data-testid="baseButton-primary"]:hover {
  background: #ef4444 !important; color: #ffffff !important;
  border-color: #ef4444 !important; box-shadow: 0 0 12px rgba(239, 68, 68, 0.3) !important;
}

/* ── Layout: centered reading column + responsiveness ─────────────────── */
.block-container {
  max-width: 940px; padding-top: 2.2rem; padding-bottom: 2rem;
  margin: 0 auto;
}
.nx-auth-wrap { max-width: 440px; margin: 4rem auto 0; padding: 0 1rem; }

/* Tablets */
@media (max-width: 900px) {
  .block-container { max-width: 100%; padding-left: 1rem; padding-right: 1rem; }
}
/* Phones */
@media (max-width: 640px) {
  .block-container { padding-top: 1.2rem; padding-left: 0.7rem; padding-right: 0.7rem; }
  .nx-header h2 { font-size: 1.3rem; }
  .nx-title { font-size: 1.2rem; }
  [data-testid="stChatMessage"] { padding: 0.95rem 1.05rem; border-radius: 12px; }
  [data-testid="stChatMessageContent"] { font-size: 0.9rem; }
  .nx-empty { padding: 2.5rem 1rem; }
  div[data-testid="stChatInput"] textarea { font-size: 0.95rem; }
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

def brand(sub: str) -> str:
    return (
        f'<div class="nx-brand">{LOGO_HTML}'
        f'<div><div class="nx-title">{APP_NAME}</div>'
        f'<div class="nx-sub">{sub}</div></div></div>'
    )


def log_and_msg(friendly: str, exc: Exception, where: str) -> str:
    """Log the real error to the server console; return a clean user message.

    Keeps raw exceptions (hosts, tracebacks, keys) OUT of the browser while
    still leaving the developer something to debug in the terminal.
    """
    print(f"[ZenRag:{where}] {type(exc).__name__}: {exc}", file=sys.stderr)
    return friendly


# --------------------------------------------------------------------------- #
# Cached pipeline + helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_pipeline(
    store_path: str, use_hyde: bool, persona: str, allow_general: bool
) -> RAGPipeline:
    return RAGPipeline.from_store(
        store_path, use_hyde=use_hyde, persona=persona, allow_general=allow_general
    )


def index_ready(store_path: str) -> bool:
    if PG:
        try:
            return len(open_for_read(settings)) > 0
        except Exception:
            return False
    return Path(store_path).exists()


def _all_tenant_ids() -> list[str]:
    """Every distinct tenant_id present in the store (full, unscoped ids)."""
    ids: list[str] = []
    try:
        if PG:
            store = open_for_read(settings)
            if hasattr(store, "_conn"):
                with store._conn.cursor() as cur:
                    cur.execute("SELECT to_regclass(%s)", (f"public.{store._table}",))
                    if cur.fetchone()[0] is not None:
                        cur.execute(f"SELECT DISTINCT tenant_id FROM {store._table}")
                        ids = [r[0] for r in cur.fetchall() if r[0]]
        elif Path(settings.store_path).exists():
            store = open_for_read(settings)
            ids = [c.metadata.get("tenant_id") for c in store._chunks]
            ids = [t for t in ids if t]
    except Exception:
        pass
    return ids


def get_workspaces() -> list[str]:
    """Workspaces visible to the current account (short names)."""
    workspaces = {"default"}
    for tid in _all_tenant_ids():
        if USER_PREFIX:
            if tid.startswith(USER_PREFIX):
                workspaces.add(tid[len(USER_PREFIX):] or "default")
        else:
            workspaces.add(tid)
    return sorted(workspaces)


def rename_workspace(old_full: str, new_full: str) -> None:
    if not old_full or not new_full or old_full == new_full:
        return
    try:
        store = open_for_write(settings, append=True)
        if PG and hasattr(store, "_conn"):
            with store._conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {store._table} SET tenant_id = %s WHERE tenant_id = %s",
                    (new_full, old_full),
                )
        elif not PG:
            for c in store._chunks:
                if c.metadata.get("tenant_id") == old_full:
                    c.metadata["tenant_id"] = new_full
            store.save(settings.store_path)
    except Exception:
        pass


def delete_workspace(full: str) -> None:
    if not full:
        return
    try:
        store = open_for_write(settings, append=True)
        if PG and hasattr(store, "_conn"):
            with store._conn.cursor() as cur:
                cur.execute(f"DELETE FROM {store._table} WHERE tenant_id = %s", (full,))
        elif not PG:
            keep = [i for i, c in enumerate(store._chunks)
                    if c.metadata.get("tenant_id") != full]
            if keep and store._vectors is not None:
                store._chunks = [store._chunks[i] for i in keep]
                store._vectors = store._vectors[keep]
            else:
                store._chunks, store._vectors = [], None
            store.save(settings.store_path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Auth gate (only when Supabase Auth is configured)
# --------------------------------------------------------------------------- #
def login_screen() -> None:
    st.markdown('<div class="nx-auth-wrap">', unsafe_allow_html=True)
    st.markdown(brand("Sign in to your workspace"), unsafe_allow_html=True)
    tab_in, tab_up = st.tabs(["Sign in", "Create account"])
    with tab_in:
        with st.form("signin"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", use_container_width=True):
                try:
                    user = auth.sign_in(email, pw)
                    st.session_state.user = {"id": user.id, "email": user.email}
                    st.rerun()
                except auth.AuthError as exc:
                    st.error(str(exc))  # already user-friendly
                except Exception as exc:
                    st.error(log_and_msg(
                        "Sign-in is unavailable right now. Please try again.",
                        exc, "signin"))
    with tab_up:
        with st.form("signup"):
            email2 = st.text_input("Email", key="su_email")
            pw2 = st.text_input("Password", type="password", key="su_pw")
            if st.form_submit_button("Create account", use_container_width=True):
                try:
                    auth.sign_up(email2, pw2)
                    st.success("Account created. If email confirmation is on, "
                               "confirm it, then sign in.")
                except auth.AuthError as exc:
                    st.error(str(exc))  # already user-friendly
                except Exception as exc:
                    st.error(log_and_msg(
                        "Couldn't create the account right now. Please try again.",
                        exc, "signup"))
    st.markdown("</div>", unsafe_allow_html=True)


if settings.auth_enabled and "user" not in st.session_state:
    login_screen()
    st.stop()

# Effective account scope: signed-in user's id (isolated), or none when auth off.
current_user = st.session_state.get("user")
USER_PREFIX = f"{current_user['id']}/" if current_user else ""


def full_tenant(workspace: str) -> str:
    return f"{USER_PREFIX}{workspace}"


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
    st.markdown(brand("Semantic Search · Grounded Answers"), unsafe_allow_html=True)

    if current_user:
        st.markdown(f'<div class="nx-account">{current_user["email"]}</div>',
                    unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            for k in ("user", "messages", "workspaces"):
                st.session_state.pop(k, None)
            st.rerun()

    # ── Workspace selector (switch / new / rename / delete)
    st.markdown('<div class="nx-section">Workspace</div>', unsafe_allow_html=True)

    workspaces = list(st.session_state.workspaces)
    if st.session_state.workspace not in workspaces:
        workspaces = sorted(set(workspaces) | {st.session_state.workspace})
        st.session_state.workspaces = workspaces

    with st.expander(st.session_state.workspace, expanded=False):
        st.markdown("**Switch workspace**")
        for ws in workspaces:
            label = f"• {ws}" if ws == st.session_state.workspace else ws
            if st.button(label, key=f"ws_sel_{ws}", use_container_width=True):
                st.session_state.workspace = ws
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")
        tab_new, tab_rename, tab_delete = st.tabs(["New", "Rename", "Delete"])

        with tab_new:
            with st.form("add_ws_form", clear_on_submit=True):
                new_ws = st.text_input("Workspace name", placeholder="e.g. client-abc")
                if st.form_submit_button("Create workspace", use_container_width=True):
                    new_ws = new_ws.strip().lower().replace(" ", "-")
                    if new_ws:
                        st.session_state.workspace = new_ws
                        st.session_state.workspaces = get_workspaces()
                        st.rerun()

        with tab_rename:
            st.markdown(f"Rename current: `{st.session_state.workspace}`")
            new_name = st.text_input("New name", value=st.session_state.workspace,
                                     key="popover_rename")
            if st.button("Save", key="popover_rename_save", use_container_width=True):
                new_name = new_name.strip().lower().replace(" ", "-")
                if new_name and new_name != st.session_state.workspace:
                    rename_workspace(full_tenant(st.session_state.workspace),
                                     full_tenant(new_name))
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
                if st.button("Delete workspace", key="popover_delete",
                             use_container_width=True, type="primary"):
                    delete_workspace(full_tenant(st.session_state.workspace))
                    st.session_state.workspace = "default"
                    load_pipeline.clear()
                    st.session_state.workspaces = get_workspaces()
                    st.rerun()

    tenant_id = full_tenant(st.session_state.workspace)

    # ── Persona
    st.markdown('<div class="nx-section">Persona</div>', unsafe_allow_html=True)
    persona = st.text_area(
        "Bot personality (optional)", value=settings.bot_persona,
        placeholder="e.g. You are Ava, a warm, concise support agent for Acme.",
        height=68, label_visibility="collapsed",
    )

    # ── Knowledge base
    st.markdown('<div class="nx-section">Knowledge Base</div>', unsafe_allow_html=True)
    with st.expander("Add sources", expanded=not index_ready(settings.store_path)):
        uploads = st.file_uploader(
            "Upload docs (.pdf .txt .md .csv .xlsx)",
            type=["pdf", "txt", "md", "markdown", "rst", "csv", "xlsx"],
            accept_multiple_files=True,
        )
        url = st.text_input("Add a single web page", placeholder="https://example.com/faq")
        crawl_url = st.text_input("Crawl a whole site", placeholder="https://example.com")
        crawl_pages = st.slider("Max pages to crawl", 1, 50, 15)
        use_samples = st.checkbox("Include sample docs", value=False)
        append = st.checkbox("Append to existing", value=True)

        st.markdown('<div class="build-btn">', unsafe_allow_html=True)
        build_clicked = st.button("Build knowledge base", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if build_clicked:
            paths, urls, crawl_urls = [], [], []
            tmpdir = Path(tempfile.mkdtemp(prefix="nx_upload_"))
            for f in uploads or []:
                dest = tmpdir / f.name
                dest.write_bytes(f.getbuffer())
                paths.append(str(dest))
            if use_samples:
                paths.append("data/sample_docs")
            if url.strip():
                urls.append(url.strip())
            if crawl_url.strip():
                crawl_urls.append(crawl_url.strip())

            if not paths and not urls and not crawl_urls:
                st.error("Upload a file, add/crawl a URL, or tick sample docs.")
            else:
                try:
                    with st.spinner("Fetching, embedding & indexing…"):
                        store = open_for_write(settings, append=append, tenant_id=tenant_id)
                        build_store(
                            paths, urls=urls, crawl_urls=crawl_urls,
                            crawl_max_pages=crawl_pages, store=store, tenant_id=tenant_id,
                        )
                        store.save(settings.store_path)
                    load_pipeline.clear()
                    st.session_state.workspaces = get_workspaces()
                    st.success(f"Knowledge base updated — {len(store)} chunks indexed.")
                except Exception as exc:
                    st.error(log_and_msg(
                        "Couldn't add those sources. Check your data / connection "
                        "settings and try again.", exc, "ingest"))

    # ── Retrieval settings
    st.markdown('<div class="nx-section">Retrieval</div>', unsafe_allow_html=True)
    with st.expander("Settings", expanded=False):
        top_k = st.slider("Sources per answer", 1, 15, settings.top_k)
        use_hyde = st.toggle("HyDE query rewriting", value=False)
        allow_general = st.toggle(
            "Answer from general knowledge if not in docs", value=False,
            help="On: falls back to the model's own knowledge (labelled) when your "
                 "documents don't cover the question. Off: strictly grounded + cited.",
        )
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
    f'<div class="nx-header"><h2>{APP_NAME}</h2>'
    f'<div class="nx-header-sub">Workspace: <strong>{st.session_state.workspace}</strong> · '
    f"Ask anything about your uploaded knowledge base</div></div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="nx-empty"><h3>Ready to chat</h3>'
        "<p>Add sources in the sidebar, build your knowledge base, "
        "then ask a question below.</p></div>",
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
            rag = load_pipeline(settings.store_path, use_hyde, persona, allow_general)
            tokens, answer = rag.stream_query(
                prompt, top_k=top_k, tenant_id=tenant_id, history=history
            )
            streamed = ""
            for delta in tokens:
                streamed += delta
                placeholder.markdown(streamed + "▌")
            placeholder.markdown(streamed)
        except Exception as exc:
            placeholder.error(log_and_msg(
                "Sorry — I couldn't generate a response just now. Please try again.",
                exc, "chat"))
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
