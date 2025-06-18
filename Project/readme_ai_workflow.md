# AI Voice Memo Workflow

A modular Python system that processes voice memos through Azure Speech-to-Text and Azure OpenAI to generate structured medical summaries.

## 🎯 System Overview

This workflow transforms voice recordings into structured medical reports through three main steps:

```
Voice Memo → Speech-to-Text → AI Summarization → Structured Output
```

### Architecture Flow
1. **Audio Input**: Voice memo file (WAV, MP3, etc.)
2. **Transcription**: Azure Speech-to-Text API converts audio to text
3. **Summarization**: Azure OpenAI (GPT-4o) processes transcript into structured medical format
4. **Output**: Formatted medical summary with standardized fields

## 📁 Project Structure

```
├── ai_workflow.py          # Main workflow functions
├── config.py              # Azure credentials configuration
└── readme_ai_workflow.md   # This documentation
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.7+
- Azure Cognitive Services account
- Azure OpenAI account

### 1. Install Dependencies
```bash
pip install openai requests
```

### 2. Configure Azure Credentials
Edit `config.py` with your actual Azure keys:

```python
# Azure Speech-to-Text
AZURE_SPEECH_KEY = "your_speech_key_here"
AZURE_SPEECH_ENDPOINT = "https://your-endpoint.cognitiveservices.azure.com/"

# Azure OpenAI
AZURE_OPENAI_API_KEY = "your_openai_key_here"
AZURE_OPENAI_ENDPOINT = "https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
```

### 3. Test the System
```python
from ai_workflow import process_voice_memo

result = process_voice_memo("path/to/your/audio.wav")
print(result["summary"])
```

## 🚀 Usage Examples

### Basic Usage
```python
from ai_workflow import process_voice_memo

# Process a voice memo
result = process_voice_memo("medical_recording.wav")

if result["status"] == "success":
    print("Transcript:", result["transcript"])
    print("Summary:", result["summary"])
else:
    print("Error:", result["error"])
```

### Individual Functions
```python
from ai_workflow import transcribe_audio, summarize_transcript

# Step 1: Transcribe audio
transcript = transcribe_audio("audio.wav", locale="en-US")

# Step 2: Summarize transcript
summary = summarize_transcript(transcript)
```

### Multilingual Support
```python
# Spanish transcription
result = process_voice_memo("audio_spanish.wav", locale="es-ES")

# French transcription
result = process_voice_memo("audio_french.wav", locale="fr-FR")
```

## 📱 Mobile App Integration

### Architecture for Mobile Apps

```
Mobile App Frontend
        ↓
Mobile App Backend (API)
        ↓
AI Workflow Module
        ↓
Azure Services
```

### Integration Options

#### Option 1: Direct Integration (Python Backend)
```python
# app.py (Flask/FastAPI backend)
from flask import Flask, request, jsonify
from ai_workflow import process_voice_memo
import os

app = Flask(__name__)

@app.route('/process-voice-memo', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Save temporarily
    temp_path = f"temp_{audio_file.filename}"
    audio_file.save(temp_path)

    try:
        # Process the audio
        result = process_voice_memo(temp_path)

        # Clean up
        os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

#### Option 2: Microservice Architecture
```python
# microservice.py
from fastapi import FastAPI, UploadFile, File
from ai_workflow import process_voice_memo
import tempfile
import os

app = FastAPI()

@app.post("/transcribe-and-summarize")
async def process_voice_memo_endpoint(audio: UploadFile = File(...)):
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        result = process_voice_memo(temp_path)
        return result
    finally:
        os.unlink(temp_path)
```

### Mobile App Implementation Examples

#### React Native
```javascript
// VoiceMemoService.js
export const processVoiceMemo = async (audioUri) => {
  const formData = new FormData();
  formData.append('audio', {
    uri: audioUri,
    type: 'audio/wav',
    name: 'voice_memo.wav',
  });

  try {
    const response = await fetch('YOUR_API_ENDPOINT/process-voice-memo', {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return await response.json();
  } catch (error) {
    console.error('Error processing voice memo:', error);
    throw error;
  }
};
```

#### Flutter
```dart
// voice_memo_service.dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class VoiceMemoService {
  static Future<Map<String, dynamic>> processVoiceMemo(String audioPath) async {
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('YOUR_API_ENDPOINT/process-voice-memo')
    );

    request.files.add(await http.MultipartFile.fromPath('audio', audioPath));

    var response = await request.send();
    var responseData = await response.stream.bytesToString();

    return json.decode(responseData);
  }
}
```

#### Swift (iOS)
```swift
// VoiceMemoService.swift
import Foundation

class VoiceMemoService {
    static func processVoiceMemo(audioURL: URL, completion: @escaping (Result<[String: Any], Error>) -> Void) {
        let url = URL(string: "YOUR_API_ENDPOINT/process-voice-memo")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let audioData = try! Data(contentsOf: audioURL)
        let body = createMultipartBody(boundary: boundary, audioData: audioData)
        request.httpBody = body

        URLSession.shared.dataTask(with: request) { data, response, error in
            // Handle response
        }.resume()
    }
}
```

## 📋 Output Format

The system generates structured medical summaries with these fields:

```
Visit_Date: [Date of visit]
Chief_Complaint: [Main patient concern]
Diagnosis_ICD10: [ICD-10 code if available]
Diagnosis_Label: [Diagnosis description]
Diagnosis_Stage: [Stage/severity if applicable]
Symptoms: [List of symptoms]
Exam_Findings: [Physical examination results]
Investigations: [Previous tests/studies]
Tests_Ordered: [New tests ordered]
Plan_Therapy: [Treatment plan]
Plan_Medications: [Prescribed medications]
Plan_Assistive: [Assistive devices/aids]
Plan_Follow_Up: [Follow-up instructions]
```

## 🔒 Security Considerations

### For Production Deployment:
1. **API Keys**: Store in environment variables or secure key management
2. **File Handling**: Implement secure temporary file handling
3. **Input Validation**: Validate audio file types and sizes
4. **Rate Limiting**: Implement API rate limiting
5. **Logging**: Add comprehensive logging for debugging
6. **Error Handling**: Implement robust error handling

### Example Secure Configuration:
```python
# secure_config.py
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

# Validate required environment variables
required_vars = ['AZURE_SPEECH_KEY', 'AZURE_OPENAI_API_KEY']
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")
```

## 🐛 Troubleshooting

### Common Issues:

1. **Authentication Errors**
   - Verify Azure keys are correct
   - Check endpoint URLs
   - Ensure services are active

2. **Audio Format Issues**
   - Supported formats: WAV, MP3, M4A, FLAC
   - Recommended: WAV, 16kHz, mono

3. **Transcription Errors**
   - Check audio quality
   - Verify locale setting
   - Ensure audio is audible

4. **API Response Issues**
   - Check network connectivity
   - Verify API quotas
   - Review error logs

## 📈 Performance Optimization

### For High-Volume Applications:
1. **Async Processing**: Use async/await for concurrent requests
2. **Caching**: Cache frequently used responses
3. **Batch Processing**: Process multiple files together
4. **Connection Pooling**: Reuse HTTP connections
5. **Monitoring**: Implement performance monitoring

## 🔄 Future Enhancements

- [ ] Real-time streaming transcription
- [ ] Speaker diarization (multiple speakers)
- [ ] Custom medical vocabulary
- [ ] Integration with EHR systems
- [ ] Multi-language medical terminology
- [ ] Voice activity detection
- [ ] Noise reduction preprocessing

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Azure service status
3. Validate configuration settings
4. Check API quotas and limits

## 📄 License

This project is designed for educational and development purposes. Ensure compliance with healthcare regulations (HIPAA, GDPR) when handling medical data in production environments.
