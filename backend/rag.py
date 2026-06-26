"""
RAG layer for the Resume Interview feature.

Pipeline:
    upload (PDF/DOCX) -> extract text -> chunk -> embed (Gemini) -> store (Chroma)
    interview start   -> retrieve chunks across fixed query angles -> build context block

Each resume upload gets its own Chroma collection (named by session_id) so multiple
people can upload resumes without clobbering each other, and re-uploading just
recreates the collection for that session.
"""

import os
import uuid
import shutil
from typing import List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types as genai_types
from pypdf import PdfReader
import docx

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = "gemini-embedding-2"

# Persistent client - collections live on disk under CHROMA_DIR
_client = chromadb.PersistentClient(path=CHROMA_DIR)

_genai_client = genai.Client(api_key=GOOGLE_API_KEY)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom Chroma embedding function using the modern `google-genai` SDK.
    (Chroma's built-in GoogleGenerativeAiEmbeddingFunction depends on the
    deprecated `google-generativeai` package, which has version conflicts
    with newer client libraries - so we implement this directly instead.)
    """

    def __init__(self, task_type="retrieval_document"):
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        # Process documents in batches if needed, but return one embedding per input document
        embeddings = []
        for doc in input:
            result = _genai_client.models.embed_content(
                model=EMBED_MODEL,
                contents=doc,
                config=genai_types.EmbedContentConfig(task_type=self.task_type),
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


_embedding_fn = GeminiEmbeddingFunction(task_type="retrieval_document")
_query_embedding_fn = GeminiEmbeddingFunction(task_type="retrieval_query")

# Fixed retrieval angles - run once at interview start to pull a well-rounded
# slice of the resume rather than relying on a single generic query.
RETRIEVAL_QUERIES = [
    "work experience and job responsibilities",
    "technical skills and tools",
    "projects built and technologies used",
    "education and certifications",
    "achievements and measurable impact",
]

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between consecutive chunks


def extract_text(file_path, filename):
    """Extract raw text from an uploaded PDF or DOCX file."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError(f"Unsupported file type: .{ext}. Please upload a PDF or DOCX.")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker over raw text."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _collection_name(session_id):
    return f"resume_{session_id}"


def index_resume(file_path, filename, session_id):
    """
    Parse + chunk + embed a resume, storing it in a Chroma collection scoped
    to this session_id. Returns the number of chunks indexed.
    """
    text = extract_text(file_path, filename)
    if not text.strip():
        raise ValueError("Couldn't extract any text from that file - is it a scanned/image-only PDF?")

    chunks = chunk_text(text)

    name = _collection_name(session_id)
    # Drop any previous collection for this session (re-upload = replace)
    try:
        _client.delete_collection(name)
    except Exception:
        pass

    collection = _client.create_collection(name=name, embedding_function=_embedding_fn)
    collection.add(
        documents=chunks,
        ids=[f"{session_id}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)


def retrieve_resume_context(session_id, top_k=3):
    """
    Run the fixed set of retrieval queries against the session's resume
    collection and return a deduplicated, formatted context block ready
    to drop into a prompt.
    """
    name = _collection_name(session_id)
    try:
        collection = _client.get_collection(name=name, embedding_function=_embedding_fn)
    except Exception:
        raise ValueError("No resume found for this session - please upload one first.")

    seen = set()
    sections = []
    for query in RETRIEVAL_QUERIES:
        query_embedding = _query_embedding_fn([query])
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        docs = results.get("documents", [[]])[0]
        for doc in docs:
            if doc not in seen:
                seen.add(doc)
                sections.append(doc)

    return "\n---\n".join(sections)


def has_resume(session_id):
    try:
        _client.get_collection(name=_collection_name(session_id))
        return True
    except Exception:
        return False


def clear_resume(session_id):
    try:
        _client.delete_collection(_collection_name(session_id))
    except Exception:
        pass
