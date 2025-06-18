import os
from openai import AzureOpenAI
from rag_manager import RAGManager
from typing import List, Dict, Any

# Configuration
try:
    from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
except ImportError:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://hubtad5418503785.openai.azure.com/")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

class MedicalChatbot:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2025-01-01-preview",
        )
        self.rag = RAGManager()
        self.conversation_history = []

    def chat(self, user_message: str, patient_id: str = None) -> str:
        """
        Process user message and return AI response using RAG

        Args:
            user_message (str): User's question or message
            patient_id (str, optional): Specific patient to focus on

        Returns:
            str: AI assistant response
        """
        try:
            # Search relevant information from RAG
            relevant_docs = self._get_relevant_context(user_message, patient_id)

            # Build context from search results
            context = self._build_context(relevant_docs, patient_id)

            # Create chat prompt with RAG context
            messages = self._build_chat_messages(user_message, context)

            # Get AI response
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                max_tokens=1000,
                temperature=0.3,
            )

            ai_response = response.choices[0].message.content

            # Store conversation
            self.conversation_history.append({
                "user": user_message,
                "assistant": ai_response,
                "patient_id": patient_id
            })

            return ai_response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

    def _get_relevant_context(self, query: str, patient_id: str = None) -> List[Dict[str, Any]]:
        """Get relevant documents from RAG database"""

        # If patient_id specified, get their specific data
        if patient_id:
            patient_history = self.rag.get_patient_history(patient_id)
            search_results = self.rag.search_patient_data(query, patient_id, top=3)

            # Combine and deduplicate
            all_docs = patient_history + search_results
            seen_texts = set()
            unique_docs = []
            for doc in all_docs:
                text = doc.get('text', '')
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_docs.append(doc)

            return unique_docs[:8]  # Limit to avoid token overflow
        else:
            # General search across all patients
            return self.rag.search_patient_data(query, top=5)

    def _build_context(self, documents: List[Dict[str, Any]], patient_id: str = None) -> str:
        """Build context string from relevant documents"""

        if not documents:
            return "No relevant medical records found."

        context_parts = []

        if patient_id:
            context_parts.append(f"=== MEDICAL RECORDS FOR PATIENT: {patient_id.upper()} ===\n")
        else:
            context_parts.append("=== RELEVANT MEDICAL RECORDS ===\n")

        for doc in documents:
            patient = doc.get('patient_id', 'Unknown')
            date = doc.get('date', 'Unknown date')
            category = doc.get('category', 'General')
            text = doc.get('text', '')

            context_parts.append(f"Patient: {patient} | Date: {date} | Type: {category}")
            context_parts.append(f"Content: {text}")
            context_parts.append("---")

        return "\n".join(context_parts)

    def _build_chat_messages(self, user_message: str, context: str) -> List[Dict[str, Any]]:
        """Build the complete message array for the AI"""

        system_message = """You are a medical assistant AI with access to patient medical records. Your role is to:

1. **Answer medical questions** based on the provided patient records
2. **Provide consultation summaries** when requested
3. **Suggest next steps** based on medical history
4. **Generate briefings for doctors** when needed

**IMPORTANT GUIDELINES:**
- Always base your responses on the provided medical records
- Be professional and use appropriate medical terminology
- If information is not available in the records, clearly state this
- Never provide emergency medical advice - always recommend seeing a healthcare provider for urgent concerns
- Maintain patient confidentiality and professionalism
- Structure your responses clearly with relevant medical context

**CONTEXT FROM MEDICAL RECORDS:**
{context}

Please respond to the user's question based on this medical information."""

        messages = [
            {
                "role": "system",
                "content": system_message.format(context=context)
            }
        ]

        # Add recent conversation history (last 4 exchanges)
        recent_history = self.conversation_history[-4:] if self.conversation_history else []
        for exchange in recent_history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history"""
        return self.conversation_history

# CLI testing function
if __name__ == "__main__":
    chatbot = MedicalChatbot()

    print("=== Medical AI Chatbot Test ===")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("You can specify a patient by starting with 'patient:moayad' or similar\n")

    while True:
        user_input = input("\n🧑‍⚕️ You: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            chatbot.reset_conversation()
            print("✅ Conversation history cleared!")
            continue
        elif not user_input:
            continue

        # Check if patient specified
        patient_id = None
        if user_input.startswith('patient:'):
            parts = user_input.split(':', 1)
            if len(parts) == 2:
                patient_id = parts[0].replace('patient', '').strip()
                user_input = parts[1].strip()

        # Get AI response
        print("\n🤖 AI Doctor: ", end="")
        response = chatbot.chat(user_input, patient_id)
        print(response)

    print("\n👋 Thanks for using the Medical AI Chatbot!")
