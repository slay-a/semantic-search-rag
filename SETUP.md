# Setup Guide

End-to-end setup for the NextCX RAG chatbot: OpenAI for AI, Supabase for the
vector database **and** login, with multi-source ingest (files, URLs, whole-site
crawl), workspaces, and a light/dark chat UI.

---

## 1. Prerequisites

- **Python 3.10+**
- An **OpenAI** account (for embeddings + generation)
- A **Supabase** account (free tier) — for the vector database and login

---

## 2. Get your OpenAI key

1. Go to <https://platform.openai.com/api-keys> → **Create new secret key** → copy it.
2. Add a payment method under **Billing** (API access is pre-paid; ~$5 is plenty).
   Embeddings are ~$0.02 / million tokens; a demo costs pennies.

---

## 3. Install the project

```bash
git clone <your-repo-url>
cd semantic-search-rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 4. Create the Supabase project

1. Go to <https://supabase.com> → **New project**. Pick a name and a strong
   **database password** (save it — you'll need it in step 5).
2. Wait ~2 minutes for it to provision.

### 4a. Enable the vector extension

**Database → Extensions →** search `vector` → **enable** it.

### 4b. Get the database connection string (for pgvector)

**Project Settings → Database → Connection string → URI.**

- Use the **Session pooler** (port **5432**) or the **Direct connection** — *not*
  the Transaction pooler (6543), which doesn't play well with prepared statements.
- Replace `[YOUR-PASSWORD]` with the password from step 4.

It looks like:
```
postgresql://postgres.abcdxyz:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
```

### 4c. Get the Auth keys (for login)

**Project Settings → API:**
- **Project URL** → `SUPABASE_URL`
- **Project API keys → `anon` `public`** → `SUPABASE_ANON_KEY`

### 4d. (Demo tip) Turn off email confirmation

So new sign-ups work instantly without a confirmation email:
**Authentication → Providers → Email →** turn **off** "Confirm email" → save.
(Leave it on for a real deployment.)

---

## 5. Configure your `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENAI_API_KEY=sk-...

# Use Supabase as the vector store
RAG_STORE_BACKEND=pgvector
DATABASE_URL=postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require

# Turn on Supabase login
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_ANON_KEY=eyJ...

# Optional: give the bot a name/persona
RAG_BOT_NAME=Ava
RAG_BOT_PERSONA=You are Ava, a warm and concise customer-support agent.
```

> Leave `RAG_STORE_BACKEND`, `SUPABASE_URL`, and `SUPABASE_ANON_KEY` unset to run
> **locally with no database and no login** (a local file store + a manual
> "workspace" field). Great for a quick offline test.

---

## 6. Run it

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>.

---

## 7. Use it

1. **Sign in / create account** (if Supabase Auth is on). Each account is a
   private, isolated knowledge base.
2. In the sidebar → **Knowledge base**, add sources:
   - **Upload** PDF / TXT / MD / **CSV / Excel**
   - **Add a single web page** (paste a URL), or
   - **Crawl a whole site** (paste the root URL + set max pages)
   - Click **Build knowledge base**.
3. **Chat.** Answers stream in, grounded in your sources with `[n]` **citations**.
   When the answer isn't in your docs, the bot **abstains and offers a human
   handoff** instead of guessing.
4. **Projects** (optional): use the *Project* field to keep multiple bots under
   one account (e.g. one per client).
5. **Theme**: toggle light/dark in the sidebar.

---

## 8. Verify (optional)

```bash
pytest -q                                   # 14 offline tests, no keys needed
RAG_EMBEDDING_PROVIDER=hash pytest -q       # fully offline embeddings
```

---

## 9. Deploy

- **Docker (local, all-in-one):** `docker compose up --build` (app + pgvector).
- **Cloud:** host the container anywhere (Render, Railway, Fly.io) and point
  `DATABASE_URL` at Supabase. Set the same env vars in the host's dashboard.
- The database table (vector + full-text + `tenant_id` indexes) is created
  automatically on the first ingest — no manual SQL.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | Add it to `.env` (and restart the app). |
| `DATABASE_URL is not set` | You set `RAG_STORE_BACKEND=pgvector` but no URL. Add it, or unset the backend to run locally. |
| Sign-up "succeeded" but can't sign in | Email confirmation is on — confirm the email, or disable it (step 4d). |
| `extension "vector" does not exist` | Enable it (step 4a). |
| Prepared-statement / pooler errors | Use the **Session pooler (5432)** or direct connection, not Transaction (6543). |
| Crawler returns little text | Some sites block bots or render via JS; try adding key pages as single URLs. |
