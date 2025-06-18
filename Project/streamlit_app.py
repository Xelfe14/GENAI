import streamlit as st
import os
import tempfile
import io
from audio_recorder_streamlit import audio_recorder
from ai_workflow import process_voice_memo
from chat_interface import MedicalChatbot
from patient_summarizer import PatientSummarizer

# Configure Streamlit page
st.set_page_config(
    page_title="🏥 Personalized Health Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = MedicalChatbot()
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = PatientSummarizer()
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Available patients
PATIENTS = ["moayad", "taddeo", "santiago", "tomas"]

def main():
    # Header
    st.title("🏥 Personalized Health Hub")
    st.markdown("*AI-Powered Medical Assistant with Voice Recording & RAG Integration*")

    # Sidebar for patient selection
    with st.sidebar:
        st.header("👤 Patient Selection")
        st.markdown("*Select a patient to use across all features*")

        selected_patient = st.selectbox(
            "Choose Patient:",
            [""] + PATIENTS,
            index=0,
            help="This patient will be used in recording, chat, and summaries"
        )

        if selected_patient:
            st.session_state.current_patient = selected_patient
            st.success(f"🎯 Active Patient: **{selected_patient.upper()}**")
            st.markdown("✅ Ready for all features!")
        else:
            st.session_state.current_patient = ""
            st.warning("⚠️ Please select a patient to continue")

        st.divider()

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🎤 Record Consultation", "💬 Chat with AI Doctor", "📋 Generate Summary"])

    with tab1:
        record_consultation_tab()

    with tab2:
        chat_interface_tab()

    with tab3:
        generate_summary_tab()

def record_consultation_tab():
    st.header("🎤 Record Medical Consultation")

    if not st.session_state.current_patient:
        st.warning("⚠️ Please select a patient from the sidebar to continue.")
        return

    patient_id = st.session_state.current_patient
    st.info(f"Recording consultation for patient: **{patient_id.upper()}**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎙️ Voice Recording")

        # Audio recorder
        audio_bytes = audio_recorder(
            text="Click to record consultation",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x",
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            # Process audio button
            if st.button("🔄 Process Recording", type="primary"):
                with st.spinner("Processing audio consultation..."):
                    try:
                                                # Create a proper WAV file from audio bytes
                        import io

                        # Debug: Check audio data
                        st.info(f"🎵 Audio data: {len(audio_bytes)} bytes, Type: {type(audio_bytes)}")

                        # Check if we have enough audio data
                        if len(audio_bytes) < 1000:  # Less than 1KB might be too short
                            st.warning("⚠️ Audio recording seems very short. Try recording for longer.")

                        # Save audio to temporary file with proper headers
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            # The audio_recorder returns raw audio bytes, write them directly
                            tmp_file.write(audio_bytes)
                            tmp_file.flush()  # Ensure data is written
                            tmp_file_path = tmp_file.name

                        # Verify file exists and has content
                        file_size = os.path.getsize(tmp_file_path)
                        if os.path.exists(tmp_file_path) and file_size > 0:
                            st.info(f"📁 Audio file created: {file_size} bytes")

                            # For very short recordings, warn the user
                            if file_size < 5000:  # Less than 5KB
                                st.warning("⚠️ Audio file is quite small. If transcription fails, try recording for longer with clearer speech.")

                            # Process the audio
                            result = process_voice_memo(
                                audio_path=tmp_file_path,
                                patient_id=patient_id,
                                ingest_to_rag=True
                            )
                        else:
                            raise ValueError("Failed to create valid audio file")

                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)

                        # Display results
                        if result["status"] == "success":
                            st.session_state.last_result = result
                            st.success("✅ Consultation processed successfully!")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")

    with col2:
        st.subheader("📝 Processing Results")

        if 'last_result' in st.session_state and st.session_state.last_result:
            result = st.session_state.last_result

            # Transcript
            with st.expander("📄 Transcript", expanded=False):
                st.text_area(
                    "Transcribed Text:",
                    value=result.get("transcript", ""),
                    height=150,
                    disabled=True
                )

            # Summary
            with st.expander("📋 Medical Summary", expanded=True):
                st.text_area(
                    "AI-Generated Summary:",
                    value=result.get("summary", ""),
                    height=200,
                    disabled=True
                )

            # RAG Status
            rag_status = result.get("rag_ingested", False)
            if rag_status:
                st.success("✅ Summary added to RAG database")
            else:
                st.warning("⚠️ Summary not added to RAG database")



def chat_interface_tab():
    st.header("Chat with AI Doctor")

    if not st.session_state.current_patient:
        st.warning("⚠️ Please select a patient from the sidebar to continue.")
        return

    # Use global patient selection
    patient_filter = st.session_state.current_patient

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"💬 Chatting about patient: **{patient_filter.upper()}**")
    with col2:
        if st.button("🔄 Reset Chat"):
            st.session_state.chatbot.reset_conversation()
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Chat history display
    st.subheader("💬 Conversation")

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Ask me about medical history, symptoms, treatments...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Consulting medical records..."):
                response = st.session_state.chatbot.chat(user_input, patient_filter)
                st.write(response)

        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Suggested questions
    if not st.session_state.chat_history:
        st.subheader("💡 Suggested Questions")
        # Dynamic suggestions based on current patient
        patient_name = patient_filter.capitalize()
        suggestions = [
            f"What is {patient_name}'s current medical status?",
            f"Tell me about {patient_name}'s recent visits",
            f"What medications is {patient_name} currently taking?",
            f"Any follow-up appointments for {patient_name}?"
        ]

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    # Process suggestion as if user typed it
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    with st.spinner("Processing..."):
                        response = st.session_state.chatbot.chat(suggestion, patient_filter)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

def generate_summary_tab():
    st.header("Generate Medical Summaries")

    if not st.session_state.current_patient:
        st.warning("⚠️ Please select a patient from the sidebar to continue.")
        return

    # Use global patient selection
    summary_patient = st.session_state.current_patient

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Summary Options")
        st.info(f"📋 Generating summary for: **{summary_patient.upper()}**")

        # Summary type
        summary_type = st.radio(
            "Summary Type:",
            ["Comprehensive Briefing", "Recent Developments", "Condition-Specific"],
            index=0
        )

        # Condition input (only for condition-specific)
        condition = ""
        if summary_type == "Condition-Specific":
            condition = st.text_input(
                "Condition/Symptom:",
                placeholder="e.g., cholesterol, asthma, appendicitis"
            )

        # Generate button
        generate_button = st.button("📋 Generate Summary", type="primary")

    with col2:
        st.subheader("📄 Generated Summary")

        if generate_button:
            if summary_type == "Condition-Specific" and not condition:
                st.error("Please enter a condition for condition-specific summary.")
            else:
                with st.spinner(f"Generating {summary_type.lower()} for {summary_patient}..."):
                    try:
                        if summary_type == "Comprehensive Briefing":
                            summary = st.session_state.summarizer.generate_doctor_briefing(
                                summary_patient, "comprehensive"
                            )
                        elif summary_type == "Recent Developments":
                            summary = st.session_state.summarizer.generate_doctor_briefing(
                                summary_patient, "recent"
                            )
                        else:  # Condition-Specific
                            summary = st.session_state.summarizer.generate_condition_summary(
                                summary_patient, condition
                            )

                        # Display as rendered markdown
                        st.markdown(f"**{summary_type} for Patient: {summary_patient.upper()}**")
                        st.markdown("---")
                        st.markdown(summary)

                        # Download button
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name=f"{summary_patient}_{summary_type.lower().replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")

        else:
            st.info("👆 Configure options and click 'Generate Summary' to create a medical briefing.")

if __name__ == "__main__":
    main()
