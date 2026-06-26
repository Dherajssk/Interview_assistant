from flask import Flask,request,jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import assemblyai as aai
import os
import base64
import requests
import tempfile
import json
import rag

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = ASSEMBLYAI_API_KEY
checkpointer = InMemorySaver()

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer
)

question_count = 0
current_subject = ""
thread_id = "interview_session"
resume_session_id = "default_resume_session"  # single-user local app - one resume at a time

INTERVIEW_PROMPT = """You are Natalie, a friendly and conversational interviewer conducting a natural {subject} interview.

IMPORTANT GUIDELINES:
1. Ask exactly 5 questions total throughout the interview
2. Keep questions SHORT and CRISP (1-2 sentences maximum)
3. ALWAYS reference what the candidate ACTUALLY said in their previous answer - do NOT make up or assume their answers
4. Show genuine interest with brief acknowledgments based on their REAL responses
5. Adapt questions based on their ACTUAL responses - go deeper if they're strong, adjust if uncertain
6. Be warm and conversational but CONCISE
7. No lengthy explanations - just ask clear, direct questions

CRITICAL: Read the conversation history carefully. Only acknowledge what the candidate truly said, not what you think they might have said.

Keep it short, conversational, and adaptive!"""

RESUME_PROMPT = """You are Natalie, a friendly and conversational interviewer conducting a resume-based interview.

Here are relevant excerpts from the candidate's resume:
---
{resume_context}
---

IMPORTANT GUIDELINES:
1. Ask exactly 5 questions total throughout the interview
2. Base EVERY question on something SPECIFIC from the resume excerpts above - a project, a role, a skill, a tool, a metric. Quote or reference it directly (e.g. "I see you worked on X at Y - tell me more about...")
3. Keep questions SHORT and CRISP (1-2 sentences maximum)
4. ALWAYS reference what the candidate ACTUALLY said in their previous answer - do NOT make up or assume their answers
5. Go deeper into the same resume item if their answer is strong (ask for specifics, metrics, challenges); move to a different resume item if they seem unsure
6. Do NOT repeat a resume item you've already asked about
7. Be warm and conversational but CONCISE
8. No lengthy explanations - just ask clear, direct questions

CRITICAL: Only ask about things that actually appear in the resume excerpts above. Do not invent experience the candidate doesn't have.

Keep it short, conversational, and adaptive!"""

FEEDBACK_PROMPT = """Based on our complete interview conversation, provide detailed feedback.
IMPORTANT: You MUST respond with ONLY a valid JSON object. No other text before or after.
Address the candidate directly using "you" and "your" (e.g., "You explained..." not "The candidate explained...").
Respond with ONLY this JSON structure (no markdown, no code blocks, no extra text):
{{
    "subject": "{subject}",
    "candidate_score": <1-5>,
    "feedback": "<detailed strengths with specific examples from their ACTUAL answers>",
    "areas_of_improvement": "<constructive suggestions based on gaps you noticed>"
}}
Be specific - reference ACTUAL things they said during the interview."""


app = Flask(__name__)
CORS(app, expose_headers=['X-Question-Number'])

def stream_audio(text):
    BASE_URL = "https://global.api.murf.ai/v1/speech/stream"
    payload = {
        "text": text,
        "voiceId": "en-US-natalie",
        "model": "FALCON",
        "multiNativeLocale": "en-US",
        "sampleRate": 24000,
        "format": "MP3",
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }
    response = requests.post(
        BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        stream=True
    )
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            yield base64.b64encode(chunk).decode("utf-8") + "\n"



@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    """Parse, chunk, and embed an uploaded resume (PDF or DOCX) into Chroma."""
    if "resume" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    resume_file = request.files["resume"]
    filename = resume_file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "docx"):
        return jsonify({"success": False, "error": "Only PDF or DOCX files are supported"}), 400

    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}").name
    resume_file.save(temp_path)

    try:
        num_chunks = rag.index_resume(temp_path, filename, resume_session_id)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(f"[Resume Upload Error] {e}")
        return jsonify({"success": False, "error": "Failed to process resume"}), 500
    finally:
        os.unlink(temp_path)

    print(f"[Resume Indexed] {filename} -> {num_chunks} chunks")
    return jsonify({"success": True, "chunks_indexed": num_chunks})


@app.route("/start-interview", methods=["POST"])
def start_interview():
    global question_count, current_subject, checkpointer, agent
    data = request.json
    current_subject = data.get("subject", "Python")
    question_count = 1
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer
    )
    config = {"configurable": {"thread_id": thread_id}}

    if current_subject == "Resume":
        if not rag.has_resume(resume_session_id):
            return jsonify({"success": False, "error": "Please upload a resume first"}), 400
        resume_context = rag.retrieve_resume_context(resume_session_id)
        formatted_prompt = RESUME_PROMPT.format(resume_context=resume_context)
        start_message = "Start the interview with a warm greeting and ask the first question based on something specific from the resume. Keep it SHORT (1-2 sentences)."
    else:
        formatted_prompt = INTERVIEW_PROMPT.format(subject=current_subject)
        start_message = f"Start the interview with a warm greeting and ask the first question about {current_subject}. Keep it SHORT (1-2 sentences)."

    response = agent.invoke({
        "messages": [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": start_message}
        ]
    }, config=config)
    question = response["messages"][-1].content
    print(f"\n[Question {question_count}] {question}")
    return stream_audio(question), {"Content-Type": "text/plain"}

def speech_to_text(audio_path):
  """Convert audio file to text using AssemblyAI"""
  transcriber = aai.Transcriber()
  config = aai.TranscriptionConfig(
        speech_models=["universal-3-pro", "universal-2"],
        language_detection=True, speaker_labels=True,
    )
  transcript = transcriber.transcribe(audio_path, config=config)
  return transcript.text if transcript.text else ""



@app.route("/submit-answer", methods=["POST"])
def submit_answer():
    global question_count
    audio_file = request.files["audio"]
    temp_path = (
    tempfile.NamedTemporaryFile(
      delete=False,
      suffix=".webm"
     ).name
    )
    audio_file.save(temp_path)
    answer = speech_to_text(temp_path)
    os.unlink(temp_path)
    if not answer:
        answer = "Empty Text received"
    print(f"[Answer {question_count}] {answer}")
    config = {"configurable": {"thread_id": thread_id}}
   
    agent.invoke({"messages": [{"role": "user", "content": answer}]}, config=config)


    question_count += 1
    prompt = f"""The candidate just answered question {question_count - 1}.
 
    Look at their ACTUAL answer above. Do NOT assume or make up what they said.
    
    Now ask question {question_count} of 5:
    1. Briefly acknowledge what they ACTUALLY said (1 sentence) - quote their exact words if needed
    2. Ask your next question that builds on their REAL response (1-2 sentences)
    3. If they said "I don't know" or gave a wrong answer, acknowledge that and ask something simpler
    4. Keep the TOTAL response under 3 sentences
    
    Be conversational but CONCISE. Only reference what they truly said."""
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    question = response["messages"][-1].content
    print(f"\n[Question {question_count}] {question}")
    return (stream_audio(question),
        {
        'Content-Type': 'text/plain',
        'X-Question-Number': str(question_count)
        }
    )

@app.route("/get-feedback", methods=["POST"])
def get_feedback():
    """Generate detailed interview feedback"""
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke({
        "messages": [
        {
            "role": "user", 
            "content": f"{FEEDBACK_PROMPT}\n\nReview our complete {current_subject} interview conversation and provide detailed feedback."
        }
        ]
    }, config=config)
    text = response["messages"][-1].content
    print(f"\n[Feedback Generated]\n{text}\n")
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1].replace("json", "").strip()
    feedback = json.loads(cleaned)

    return jsonify({"success": True, "feedback": feedback})


app.run(debug=True, port=5000)