import os
import requests
from openai import AzureOpenAI

# --- CONFIGURATION ---
try:
    from config import (
        AZURE_SPEECH_KEY,
        AZURE_SPEECH_ENDPOINT,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT
    )
    AZURE_OPENAI_KEY = AZURE_OPENAI_API_KEY
except ImportError:
    # Fallback to environment variables
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "YOUR_SPEECH_KEY")
    AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT", "https://hubtad5418503785.cognitiveservices.azure.com/")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://hubtad5418503785.openai.azure.com/")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

# --- 1. SPEECH TO TEXT ---
def transcribe_audio(audio_path, locale="en-US"):
    """
    Transcribe audio file using Azure Speech-to-Text API

    Args:
        audio_path (str): Path to the audio file
        locale (str): Language locale (default: "en-US")

    Returns:
        str: Transcribed text
    """
    url = f"{AZURE_SPEECH_ENDPOINT}speechtotext/transcriptions:transcribe?api-version=2024-11-15"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Accept": "application/json"
    }

    # Prepare the definition exactly as shown in Azure playground
    definition = f'{{"locales":["{locale}"],"profanityFilterMode":"Masked","channels":[0,1]}}'

    # Check if file exists and has content
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    print(f"📤 Sending audio file to Azure Speech: {file_size} bytes")

    with open(audio_path, "rb") as audio_file:
        files = {
            "audio": ("audio.wav", audio_file, "audio/wav"),
            "definition": (None, definition)
        }

        print(f"🔗 Making request to: {url}")
        response = requests.post(url, headers=headers, files=files, timeout=30)

    print(f"📥 Azure Speech response: {response.status_code}")

    # Better error handling
    if response.status_code == 401:
        raise Exception("Azure Speech authentication failed - check your API key")
    elif response.status_code == 400:
        raise Exception(f"Bad request to Azure Speech: {response.text}")
    elif response.status_code != 200:
        raise Exception(f"Azure Speech error {response.status_code}: {response.text}")

    response.raise_for_status()
    result = response.json()
    print(f"📊 Azure Speech result keys: {list(result.keys())}")

    # Extract transcript from response (using actual Azure response structure)
    transcript = ""

    # Try the actual response format first
    if "combinedPhrases" in result and result["combinedPhrases"]:
        for phrase in result["combinedPhrases"]:
            if phrase.get("text", "").strip():
                transcript += phrase["text"] + " "
        transcript = transcript.strip()

    # Fallback to other possible formats
    if not transcript and "combinedRecognizedPhrases" in result and result["combinedRecognizedPhrases"]:
        transcript = result["combinedRecognizedPhrases"][0].get("display", "")

    # Final fallback
    if not transcript:
        transcript = result.get("text", "")

    print(f"📝 Final extracted transcript: '{transcript}' ({len(transcript)} chars)")
    return transcript

# --- 2. SUMMARIZE WITH AZURE OPENAI ---
def summarize_transcript(transcript):
    """
    Summarize transcript using Azure OpenAI GPT-4o

    Args:
        transcript (str): The transcribed text

    Returns:
        str: Summarized medical report
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2025-01-01-preview",
    )

    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """## Context
Your a helpful assistant, which task is to do a summarisation of the transcript of the recording of a medical appointment.

## Input
You will receive the full transcript

## Output
Output rules (strict):

Produce one contiguous text block, no JSON, no Markdown.

Use exactly the field labels below, each followed by a colon and a single space.

If a field is absent, leave the value empty (e.g. Plan_Assistive: on its own line).

Keep lists either as - item bullet lines or comma-separated—stay consistent.

Add also a short paragraph, with a natural language summary of the transcript

Field order (write them exactly):
Visit_Date:
Chief_Complaint:
Diagnosis_Stage:
Symptoms:
Exam_Findings:
Investigations:
Tests_Ordered:
Plan_Therapy:
Plan_Medications:
Plan_Assistive:
Plan_Follow_Up:

Process hints:
• Skim full transcript, extract only problem-specific info.
• Preserve critical numbers (doses, ROM, timelines).
• Use standard medical terminology and ICD-10 codes when explicit.
• Do not fabricate data."""
                }
            ]
        },
        {
            "role": "user",
            "content": transcript
        }
    ]

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=chat_prompt,
        max_tokens=1024,
        temperature=0.2,
    )

    return response.choices[0].message.content

# --- 3. RAG INGESTION ---
def ingest_summary_to_rag(patient_id, summary_text):
    """
    Ingest medical summary into RAG database

    Args:
        patient_id (str): Patient identifier
        summary_text (str): Medical summary to ingest

    Returns:
        bool: Success status
    """
    try:
        from rag_manager import RAGManager
        rag = RAGManager()
        return rag.ingest_summary(patient_id, summary_text, "consultation_summary")
    except Exception as e:
        print(f"❌ Error ingesting to RAG: {str(e)}")
        return False

# --- 4. MAIN WORKFLOW FUNCTION ---
def process_voice_memo(audio_path, patient_id=None, locale="en-US", ingest_to_rag=True):
    """
    Complete workflow: Audio -> Transcript -> Summary -> RAG Ingestion

    Args:
        audio_path (str): Path to the audio file
        patient_id (str, optional): Patient ID for RAG ingestion
        locale (str): Language locale for transcription
        ingest_to_rag (bool): Whether to ingest summary into RAG

    Returns:
        dict: Contains transcript, summary, and ingestion status
    """
    try:
        print("🎤 Transcribing audio...")
        transcript = transcribe_audio(audio_path, locale)

        if not transcript:
            raise ValueError("No transcript generated from audio")

        print("🤖 Generating summary...")
        summary = summarize_transcript(transcript)

        rag_ingested = False
        if ingest_to_rag and patient_id:
            print(f"📚 Ingesting summary to RAG for patient {patient_id}...")
            rag_ingested = ingest_summary_to_rag(patient_id, summary)
            if rag_ingested:
                print("✅ Summary successfully added to RAG database!")
            else:
                print("⚠️ Failed to ingest summary to RAG database")

        return {
            "transcript": transcript,
            "summary": summary,
            "rag_ingested": rag_ingested,
            "status": "success"
        }

    except Exception as e:
        return {
            "transcript": "",
            "summary": "",
            "rag_ingested": False,
            "status": "error",
            "error": str(e)
        }

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Example usage
    audio_file = "path_to_your_audio_file.wav"  # Replace with your audio file path

    result = process_voice_memo(audio_file)

    if result["status"] == "success":
        print("\n" + "="*50)
        print("TRANSCRIPT:")
        print("="*50)
        print(result["transcript"])

        print("\n" + "="*50)
        print("MEDICAL SUMMARY:")
        print("="*50)
        print(result["summary"])
    else:
        print(f"Error: {result['error']}")
