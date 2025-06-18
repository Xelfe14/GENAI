import os
from openai import AzureOpenAI
from rag_manager import RAGManager
from typing import Dict, List, Any

# Configuration
try:
    from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
except ImportError:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://hubtad5418503785.openai.azure.com/")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

class PatientSummarizer:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2025-01-01-preview",
        )
        self.rag = RAGManager()

    def generate_doctor_briefing(self, patient_id: str, briefing_type: str = "comprehensive") -> str:
        """
        Generate a comprehensive medical briefing for doctors

        Args:
            patient_id (str): Patient identifier
            briefing_type (str): Type of briefing (comprehensive, recent, specific_condition)

        Returns:
            str: Formatted medical briefing
        """
        try:
            # Get patient's complete medical history
            patient_history = self.rag.get_patient_history(patient_id)

            if not patient_history:
                return f"No medical records found for patient: {patient_id}"

            # Sort by date (most recent first)
            patient_history.sort(key=lambda x: x.get('date', ''), reverse=True)

            # Build comprehensive context
            medical_context = self._build_medical_context(patient_history, patient_id)

            # Generate briefing based on type
            if briefing_type == "comprehensive":
                briefing = self._generate_comprehensive_briefing(medical_context, patient_id)
            elif briefing_type == "recent":
                briefing = self._generate_recent_briefing(medical_context, patient_id)
            else:
                briefing = self._generate_comprehensive_briefing(medical_context, patient_id)

            return briefing

        except Exception as e:
            return f"Error generating briefing: {str(e)}"

    def generate_condition_summary(self, patient_id: str, condition: str) -> str:
        """
        Generate summary focused on a specific medical condition

        Args:
            patient_id (str): Patient identifier
            condition (str): Specific condition to focus on

        Returns:
            str: Condition-focused summary
        """
        try:
            # Search for records related to the specific condition
            condition_records = self.rag.search_patient_data(condition, patient_id, top=10)

            if not condition_records:
                return f"No records found for patient {patient_id} related to {condition}"

            # Build context for condition
            context = self._build_condition_context(condition_records, patient_id, condition)

            # Generate condition-specific summary
            summary = self._generate_condition_briefing(context, patient_id, condition)

            return summary

        except Exception as e:
            return f"Error generating condition summary: {str(e)}"

    def _build_medical_context(self, records: List[Dict[str, Any]], patient_id: str) -> str:
        """Build comprehensive medical context from patient records"""

        context_parts = [
            f"=== COMPLETE MEDICAL RECORDS FOR PATIENT: {patient_id.upper()} ===\n"
        ]

        # Organize records by category
        categories = {}
        for record in records:
            category = record.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append(record)

        # Add records by category
        for category, category_records in categories.items():
            context_parts.append(f"\n--- {category.upper().replace('_', ' ')} ---")

            for record in category_records:
                date = record.get('date', 'Unknown date')
                text = record.get('text', '')
                context_parts.append(f"Date: {date}")
                context_parts.append(f"Details: {text}")
                context_parts.append("")

        return "\n".join(context_parts)

    def _build_condition_context(self, records: List[Dict[str, Any]], patient_id: str, condition: str) -> str:
        """Build context focused on specific condition"""

        context_parts = [
            f"=== MEDICAL RECORDS FOR PATIENT: {patient_id.upper()} ===",
            f"=== FOCUS: {condition.upper()} ===\n"
        ]

        for record in records:
            date = record.get('date', 'Unknown date')
            category = record.get('category', 'general')
            text = record.get('text', '')

            context_parts.append(f"Date: {date} | Type: {category}")
            context_parts.append(f"Content: {text}")
            context_parts.append("---")

        return "\n".join(context_parts)

    def _generate_comprehensive_briefing(self, context: str, patient_id: str) -> str:
        """Generate comprehensive doctor briefing"""

        prompt = f"""Based on the complete medical records provided, generate a comprehensive medical briefing for healthcare providers about patient {patient_id}.

**MEDICAL RECORDS:**
{context}

**BRIEFING REQUIREMENTS:**
Create a structured, professional medical briefing that includes:

1. **PATIENT OVERVIEW**
   - Patient ID and basic information
   - Key medical conditions and diagnoses

2. **MEDICAL HISTORY SUMMARY**
   - Chronological overview of significant medical events
   - Current active conditions and treatments

3. **CURRENT STATUS**
   - Recent visits and findings
   - Current medications and treatments
   - Active symptoms or concerns

4. **TREATMENT PLAN & RECOMMENDATIONS**
   - Ongoing treatment protocols
   - Scheduled follow-ups
   - Recommendations for future care

5. **CLINICAL NOTES**
   - Important observations
   - Patient compliance and response to treatment
   - Any special considerations

**FORMAT:** Professional medical briefing suitable for doctor-to-doctor communication.
**TONE:** Clinical, precise, and comprehensive.
**LENGTH:** Comprehensive but concise - focus on clinically relevant information."""

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a medical documentation specialist creating professional briefings for healthcare providers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.2,
        )

        return response.choices[0].message.content

    def _generate_recent_briefing(self, context: str, patient_id: str) -> str:
        """Generate briefing focused on recent developments"""

        prompt = f"""Based on the medical records provided, generate a recent developments briefing for patient {patient_id}.

**MEDICAL RECORDS:**
{context}

**FOCUS:** Recent medical developments, current status, and immediate care needs.

Create a concise briefing covering:
1. **RECENT VISITS** (last 3 months)
2. **CURRENT SYMPTOMS/CONDITIONS**
3. **ACTIVE TREATMENTS**
4. **IMMEDIATE FOLLOW-UP NEEDS**
5. **URGENT CONSIDERATIONS**

Keep it focused on actionable, current information for immediate patient care."""

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a medical assistant creating focused briefings for immediate patient care."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.2,
        )

        return response.choices[0].message.content

    def _generate_condition_briefing(self, context: str, patient_id: str, condition: str) -> str:
        """Generate condition-specific briefing"""

        prompt = f"""Based on the medical records provided, generate a focused briefing about {condition} for patient {patient_id}.

**MEDICAL RECORDS:**
{context}

**CONDITION FOCUS:** {condition}

Create a detailed summary covering:
1. **CONDITION OVERVIEW** - Current status of {condition}
2. **TREATMENT HISTORY** - Past and current treatments for this condition
3. **PATIENT RESPONSE** - How patient has responded to treatments
4. **CURRENT MANAGEMENT** - Active treatment protocols
5. **RECOMMENDATIONS** - Next steps and considerations

Focus specifically on information related to {condition} and its management."""

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": f"You are a medical specialist creating a focused briefing about {condition}."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        return response.choices[0].message.content

# CLI testing function
if __name__ == "__main__":
    summarizer = PatientSummarizer()

    print("=== Patient Summarizer Test ===")
    print("Available patients: moayad, taddeo, santiago, tomas")

    while True:
        print("\nOptions:")
        print("1. Comprehensive briefing")
        print("2. Recent developments briefing")
        print("3. Condition-specific summary")
        print("4. Quit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "4":
            break
        elif choice in ["1", "2", "3"]:
            patient_id = input("Enter patient ID: ").strip().lower()

            if choice == "1":
                print("\n📋 Generating comprehensive briefing...\n")
                briefing = summarizer.generate_doctor_briefing(patient_id, "comprehensive")
                print("="*60)
                print("COMPREHENSIVE MEDICAL BRIEFING")
                print("="*60)
                print(briefing)

            elif choice == "2":
                print("\n📋 Generating recent developments briefing...\n")
                briefing = summarizer.generate_doctor_briefing(patient_id, "recent")
                print("="*60)
                print("RECENT DEVELOPMENTS BRIEFING")
                print("="*60)
                print(briefing)

            elif choice == "3":
                condition = input("Enter condition to focus on: ").strip()
                print(f"\n📋 Generating {condition} summary...\n")
                summary = summarizer.generate_condition_summary(patient_id, condition)
                print("="*60)
                print(f"CONDITION SUMMARY: {condition.upper()}")
                print("="*60)
                print(summary)
        else:
            print("Invalid option. Please try again.")

    print("\n👋 Thanks for using the Patient Summarizer!")
