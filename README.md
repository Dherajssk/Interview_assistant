# 🎤 AI Interview Assistant

A voice-based AI mock interview platform. Pick a topic (or upload your résumé), talk to **Natalie** — an AI interviewer powered by Google Gemini — and get real-time, voice-driven questions followed by structured feedback at the end.

![Python](https://img.shields.io/badge/Python-3.13-yellow?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green)
![Gemini](https://img.shields.io/badge/Google-Gemini-blue?logo=google)

---

## ✨ Features

- **🎙️ Voice-first interviews** — speak your answers; the app transcribes them and Natalie replies with natural speech.
- **🧠 Adaptive questioning** — a LangGraph-orchestrated agent asks 5 questions per session, building each one on what you *actually* said, not a fixed script.
- **📄 Résumé-based interviews (RAG)** — upload a PDF/DOCX résumé and Natalie asks questions grounded in your real projects, roles, and skills using a Gemini-embeddings + Chroma retrieval pipeline.
- **📚 Topic interviews** — practice Self Introduction, Generative AI, Python, English, HTML, or CSS out of the box.
- **📊 Instant feedback** — after the interview, get a score out of 5, strengths, and areas of improvement, generated from the full conversation transcript.
- **🔊 Streamed TTS audio** — Natalie's voice is streamed live via Murf AI for low-latency playback.

---

## 🏗️ How it works

```
                                  User Starts Interview
                                            │
                                            ▼
                                 Select Interview Type
                                            │
                       ┌────────────────────┴────────────────────┐
                       │                                         │
                       ▼                                         ▼
              Subject Interview                          Resume Interview
                       │                                         │
                       │                              Upload Resume (PDF/DOCX)
                       │                                         │
                       │                                Extract Resume Text
                       │                                  (pypdf / python-docx)
                       │                                         │
                       │                                  Chunk Resume Text
                       │                                (800 chars, 150 overlap)
                       │                                         │
                       │                              Generate Embeddings
                       │                              (Gemini gemini-embedding-2)
                       │                                         │
                       │                                Store in ChromaDB
                       │                              (per-session collection)
                       │                                         │
                       │                            Retrieve Relevant Chunks
                       │                          (5 fixed query angles, top-k)
                       │                                         │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                                Build Interview Context
                              (subject prompt OR resume context)
                                            │
                                            ▼
                         Gemini 2.5 Flash + LangGraph Memory
                              (create_agent + InMemorySaver,
                                  per-thread conversation state)
                                            │
                              Generates Context-Aware Question
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
              Murf AI (Text-to-Speech)              Candidate Speaks Answer
              (FALCON model, streamed MP3                      │
                 via MediaSource API)                           ▼
                        │                          Browser Records Audio (.webm)
                        ▼                                       │
              Played Back to Candidate                          ▼
                                                  AssemblyAI (Speech-to-Text)
                                                  (universal-3-pro / universal-2,
                                                   speaker labels, lang detection)
                                                                 │
                                                                 ▼
                                                  Conversation Memory Updated
                                                  (LangGraph checkpointer state)
                                                                 │
                                                                 ▼
                                                  Generate Next Question
                                                  (loops until 5 questions asked)
                                                                 │
                                                                 ▼
                                                  Compile Full Transcript
                                                                 │
                                                                 ▼
                                                  Gemini Generates Structured
                                                  Feedback (score + strengths +
                                                       improvement areas)
                                                                 │
                                                                 ▼
                                                  Display Feedback to Candidate
```

**Step-by-step:**

1. The user picks a **Subject Interview** (Python, HTML, CSS, etc.) or a **Resume Interview**.
2. For résumé interviews: the uploaded PDF/DOCX is parsed with `pypdf`/`python-docx`, split into overlapping 800-character chunks, embedded with Gemini's `gemini-embedding-2` model, and stored in a per-session **ChromaDB** collection. At interview start, 5 fixed retrieval queries (work experience, skills, projects, education, achievements) pull back the most relevant chunks.
3. Either the résumé context or the subject name is formatted into a system prompt and handed to a **LangGraph agent** (`create_agent` + `InMemorySaver` checkpointer) running **Gemini 2.5 Flash**, which generates the first context-aware question.
4. The question text is streamed to **Murf AI**, converted to speech (FALCON model), and streamed back to the browser as base64-encoded MP3 chunks, played live via the **MediaSource API**.
5. The candidate records their spoken answer in-browser (`MediaRecorder` → `.webm`) and submits it.
6. The audio is transcribed by **AssemblyAI**, the transcript is fed back into the same LangGraph thread (updating conversation memory), and the agent generates the next question — repeating steps 4–6 until 5 questions have been asked.
7. Once the interview ends, the full conversation history in the agent's memory is sent back with a feedback prompt, and Gemini returns a structured JSON object (score out of 5, feedback, areas of improvement) that's rendered in the UI.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask, Flask-CORS |
| Agent orchestration | LangChain (`create_agent`) + LangGraph (`InMemorySaver` checkpointer) |
| LLM | Google Gemini 2.5 Flash (`langchain[google-genai]`) |
| Speech-to-text | AssemblyAI (`universal-3-pro` / `universal-2`, speaker labels, language detection) |
| Text-to-speech | Murf AI (streaming, FALCON model, `en-US-natalie` voice) |
| RAG (résumé feature) | ChromaDB (persistent, on-disk) + Gemini embeddings (`gemini-embedding-2`) + `pypdf` / `python-docx` |
| Frontend | Vanilla HTML/JS, Tailwind CSS (CDN), Font Awesome |

---

## 📁 Project Structure

```
Interview_assistant/
├── backend/
│   ├── app.py              # Flask routes, agent setup, audio streaming
│   ├── rag.py               # Résumé parsing, chunking, embedding, retrieval
│   ├── requirements.txt
│   └── chroma_db/           # Persistent vector store (auto-created)
└── frontend/
    ├── index.html            # UI (topic sidebar, recorder, feedback panel)
    └── index.js               # Recording, streaming playback, API calls
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys for:
  - [Google AI Studio](https://aistudio.google.com/) → `GOOGLE_API_KEY` (used for both the Gemini chat model and Gemini embeddings)
  - [AssemblyAI](https://www.assemblyai.com/) → `ASSEMBLYAI_API_KEY`
  - [Murf AI](https://murf.ai/) → `MURF_API_KEY`

### 1. Clone and install

```bash
git clone https://github.com/Dherajssk/Interview_assistant.git
cd Interview_assistant/backend
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file inside `backend/`:

```env
GOOGLE_API_KEY=your_google_api_key
MURF_API_KEY=your_murf_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
```

> ⚠️ **Never commit your `.env` file.** See [Security Notes](#-security-notes) below — this repo currently has one checked in and it should be removed from git history before going public.

### 3. Run the backend

```bash
python app.py
```

The Flask server starts on `http://127.0.0.1:5000`.

### 4. Open the frontend

Simply open `frontend/index.html` in your browser (or serve it with any static file server). Make sure your browser has microphone permissions enabled for the page.

---

## 🎮 Usage

1. **Pick a topic** from the sidebar (or click **Resume Questions** to upload a résumé first).
2. Click **Start Interview** — Natalie greets you and asks question 1 out loud.
3. Click the mic button to record your answer, then **Submit Answer**.
4. Repeat for 5 questions total.
5. Click **End Interview** → **Get Feedback** to see your score and personalized feedback.

---

## ⚠️ Security Notes

A few things worth fixing before sharing or deploying this publicly:

- **`backend/.env` is currently committed to the repo.** Add a `.gitignore` with `.env` and `chroma_db/`, then scrub the key from git history (`git filter-repo` or BFG) and rotate all three API keys immediately, since they're exposed in the zip/repo as-is.
- **`backend/chroma_db/` is also tracked** — this is a runtime artifact (your local vector store with embedded résumé data) and shouldn't be in version control.
- The app currently uses hardcoded global state (`thread_id = "interview_session"`, a single shared `resume_session_id`) rather than per-user sessions — fine for local single-user testing, but should be replaced with per-session IDs before any multi-user deployment.

Suggested `.gitignore`:
```gitignore
.env
__pycache__/
*.pyc
backend/chroma_db/
```

---

## 🗺️ Roadmap Ideas

- [ ] Per-session thread IDs (support concurrent users)
- [ ] Persist interview history/feedback to a database
- [ ] Deploy backend (Render/Railway) + frontend (Vercel/Netlify)
- [ ] Add authentication for multi-user résumé storage
- [ ] Support more interview topics / custom topic input
- [ ] Add unit tests for `rag.py` chunking and retrieval logic

---

## 📄 License

No license file is currently included. Add one (e.g. MIT) if you intend to share or accept contributions.
