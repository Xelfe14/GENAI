import os
import tempfile
import requests
from config import AZURE_SPEECH_KEY, AZURE_SPEECH_ENDPOINT

def test_azure_speech_with_dummy_audio():
    """Test Azure Speech API with a small dummy audio file"""

    print("=== Azure Speech Debug Test ===")
    print(f"Endpoint: {AZURE_SPEECH_ENDPOINT}")
    print(f"API Key length: {len(AZURE_SPEECH_KEY)} characters")
    print(f"API Key first 10 chars: {AZURE_SPEECH_KEY[:10]}...")

    # Create a minimal WAV file header for testing
    # This creates a very basic WAV file structure
    dummy_wav_data = create_minimal_wav()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(dummy_wav_data)
        tmp_file.flush()
        audio_path = tmp_file.name

    print(f"\n📁 Created test audio file: {audio_path}")
    print(f"📊 File size: {os.path.getsize(audio_path)} bytes")

    # Test the exact same API call as in ai_workflow.py
    url = f"{AZURE_SPEECH_ENDPOINT}speechtotext/transcriptions:transcribe?api-version=2024-11-15"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Accept": "application/json"
    }

    definition = '{"locales":["en-US"],"profanityFilterMode":"Masked","channels":[0,1]}'

    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "audio": ("audio.wav", audio_file, "audio/wav"),
                "definition": (None, definition)
            }

            print(f"\n🔗 Making request to: {url}")
            print(f"📤 Headers: {headers}")
            print(f"📝 Definition: {definition}")

            response = requests.post(url, headers=headers, files=files, timeout=30)

            print(f"\n📥 Response Status: {response.status_code}")
            print(f"📋 Response Headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Response keys: {list(result.keys())}")
                print(f"📊 Full response: {result}")

                # Test transcript extraction
                if "combinedRecognizedPhrases" in result and result["combinedRecognizedPhrases"]:
                    transcript = result["combinedRecognizedPhrases"][0].get("display", "")
                    print(f"📝 Extracted transcript: '{transcript}'")
                else:
                    transcript = result.get("text", "")
                    print(f"📝 Fallback transcript: '{transcript}'")

                if not transcript:
                    print("❌ Transcript is empty!")
                else:
                    print(f"✅ Transcript found: {len(transcript)} characters")

            else:
                print(f"❌ Error Response: {response.text}")

                if response.status_code == 401:
                    print("🔑 Authentication issue - check API key")
                elif response.status_code == 400:
                    print("📋 Bad request - check audio format or parameters")
                elif response.status_code == 404:
                    print("🔍 Endpoint not found - check URL")

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"\n🧹 Cleaned up test file")

def create_minimal_wav():
    """Create a minimal WAV file with silence for testing"""
    # WAV file header for 1 second of silence at 16kHz, 16-bit, mono
    # This is a very basic WAV structure that should be recognizable
    sample_rate = 16000
    duration = 1  # 1 second
    samples = sample_rate * duration

    # WAV header
    wav_header = bytearray()
    wav_header.extend(b'RIFF')  # ChunkID
    wav_header.extend((36 + samples * 2).to_bytes(4, 'little'))  # ChunkSize
    wav_header.extend(b'WAVE')  # Format
    wav_header.extend(b'fmt ')  # Subchunk1ID
    wav_header.extend((16).to_bytes(4, 'little'))  # Subchunk1Size
    wav_header.extend((1).to_bytes(2, 'little'))   # AudioFormat (PCM)
    wav_header.extend((1).to_bytes(2, 'little'))   # NumChannels (mono)
    wav_header.extend(sample_rate.to_bytes(4, 'little'))  # SampleRate
    wav_header.extend((sample_rate * 2).to_bytes(4, 'little'))  # ByteRate
    wav_header.extend((2).to_bytes(2, 'little'))   # BlockAlign
    wav_header.extend((16).to_bytes(2, 'little'))  # BitsPerSample
    wav_header.extend(b'data')  # Subchunk2ID
    wav_header.extend((samples * 2).to_bytes(4, 'little'))  # Subchunk2Size

    # Add silence (zeros) for the audio data
    audio_data = bytearray(samples * 2)  # 2 bytes per sample for 16-bit

    return wav_header + audio_data

def test_streamlit_audio_format():
    """Test what format the Streamlit audio recorder actually produces"""
    print("\n=== Streamlit Audio Format Test ===")
    print("To test this, you need to:")
    print("1. Record some audio in Streamlit")
    print("2. Print the type and first few bytes of audio_bytes")
    print("3. Check if it's already a valid WAV file or raw audio data")

    # This would be tested in the actual Streamlit app

if __name__ == "__main__":
    test_azure_speech_with_dummy_audio()
    test_streamlit_audio_format()
