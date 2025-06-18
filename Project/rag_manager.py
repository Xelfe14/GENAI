import os
import json
import requests
from typing import List, Dict, Any

try:
    from config import AZURE_OPENAI_API_KEY
except ImportError:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Azure Search Configuration
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT", "https://healthcare-search-rag.search.windows.net")
SEARCH_KEY = os.getenv("SEARCH_KEY", "Tyg7xRfZ41XHzPf332NnWSh8ZhKqWCRlDVB380QzxMAzSeD3OGhl")
SEARCH_INDEX = os.getenv("SEARCH_INDEX_NAME", "patients")

class RAGManager:
    def __init__(self):
        self.search_endpoint = SEARCH_ENDPOINT
        self.search_key = SEARCH_KEY
        self.index_name = SEARCH_INDEX
        self.headers = {
            'Content-Type': 'application/json',
            'api-key': self.search_key
        }

    def ingest_summary(self, patient_id: str, summary_text: str, category: str = "consultation_summary") -> bool:
        """
        Ingest a new medical summary into Azure Search index

        Args:
            patient_id (str): Patient identifier
            summary_text (str): The medical summary text
            category (str): Category of the document

        Returns:
            bool: Success status
        """
        try:
            # Generate unique document ID
            import uuid
            from datetime import datetime
            doc_id = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

            # Prepare document for ingestion (adapted to existing index structure)
            document = {
                "chunk_id": doc_id,
                "parent_id": patient_id,
                "content": f"Patient: {patient_id}\nCategory: {category}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n{summary_text}",
                "title": f"{patient_id} - {category}",
                "url": "",
                "filepath": f"consultation_{patient_id}_{datetime.now().strftime('%Y%m%d')}"
            }

            # Upload to Azure Search
            url = f"{self.search_endpoint}/indexes/{self.index_name}/docs/index?api-version=2023-11-01"

            payload = {
                "value": [document]
            }

            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()

            print(f"✅ Successfully ingested summary for patient {patient_id}")
            return True

        except Exception as e:
            print(f"❌ Error ingesting summary: {str(e)}")
            return False

    def search_patient_data(self, query: str, patient_id: str = None, top: int = 5) -> List[Dict[str, Any]]:
        """
        Search the RAG database for relevant information

        Args:
            query (str): Search query
            patient_id (str, optional): Filter by specific patient
            top (int): Number of results to return

        Returns:
            List[Dict]: Search results
        """
        try:
            url = f"{self.search_endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"

            # Build search payload for actual index structure
            search_query = query
            if patient_id:
                search_query = f"{query} AND {patient_id}"

            search_payload = {
                "search": search_query,
                "top": top,
                "searchFields": "content",
                "select": "chunk_id,content,title,filepath"
            }

            response = requests.post(url, headers=self.headers, json=search_payload)
            response.raise_for_status()

            results = response.json()

            # Transform results to match expected format
            transformed_results = []
            for result in results.get("value", []):
                content = result.get("content", "")
                extracted_patient = self._extract_patient_from_content(content, patient_id)

                transformed_results.append({
                    "patient_id": extracted_patient,
                    "text": content,
                    "category": result.get("title", "general"),
                    "date": "2025-06-18"  # Default date since not in index
                })

            return transformed_results

        except Exception as e:
            print(f"❌ Error searching data: {str(e)}")
            return []

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get complete medical history for a specific patient

        Args:
            patient_id (str): Patient identifier

        Returns:
            List[Dict]: Patient's complete history
        """
        try:
            url = f"{self.search_endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"

            # Search for patient name in content
            search_payload = {
                "search": patient_id,
                "top": 50,
                "searchFields": "content",
                "select": "chunk_id,content,title,filepath"
            }

            response = requests.post(url, headers=self.headers, json=search_payload)
            response.raise_for_status()

            results = response.json()

            # Transform results to match expected format
            transformed_results = []
            for result in results.get("value", []):
                content = result.get("content", "")

                # Filter results that actually contain the patient name
                if patient_id.lower() in content.lower():
                    transformed_results.append({
                        "patient_id": patient_id,
                        "text": content,
                        "category": result.get("title", "general"),
                        "date": "2025-06-18"  # Default date since not in index
                    })

            return transformed_results

        except Exception as e:
            print(f"❌ Error retrieving patient history: {str(e)}")
            return []

    def _extract_patient_from_content(self, content: str, patient_id: str = None) -> str:
        """
        Extract patient ID from content text

        Args:
            content (str): Content text
            patient_id (str, optional): Expected patient ID

        Returns:
            str: Extracted or provided patient ID
        """
        if patient_id:
            return patient_id

        # Try to extract patient name from content
        import re
        patient_patterns = [
            r"Patient:\s*(\w+)",
            r"patient\s+(\w+)",
            r"(\w+)\s+visited",
            r"(\w+)\s+reported"
        ]

        for pattern in patient_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                found_patient = match.group(1).lower()
                if found_patient in ["moayad", "taddeo", "santiago", "tomas"]:
                    return found_patient

        return "unknown"

    def _parse_summary(self, summary_text: str) -> Dict[str, str]:
        """
        Parse structured medical summary into fields

        Args:
            summary_text (str): Raw summary text

        Returns:
            Dict: Parsed fields
        """
        fields = {}

        # Extract key medical fields from the summary
        field_mappings = {
            "Visit_Date": "visit_date",
            "Chief_Complaint": "chief_complaint",
            "Diagnosis_ICD10": "diagnosis_icd10",
            "Diagnosis_Label": "diagnosis_label",
            "Symptoms": "symptoms",
            "Plan_Medications": "medications",
            "Plan_Follow_Up": "follow_up"
        }

        for field_key, field_name in field_mappings.items():
            # Simple extraction - look for field followed by colon
            import re
            pattern = f"{field_key}:\\s*(.+?)(?=\\n[A-Z_]+:|$)"
            match = re.search(pattern, summary_text, re.MULTILINE | re.DOTALL)
            if match:
                fields[field_name] = match.group(1).strip()

        return fields

# CLI testing function
if __name__ == "__main__":
    rag = RAGManager()

    print("=== RAG Manager Test ===")

    # Test 1: Search existing data
    print("\n1. Testing search functionality...")
    results = rag.search_patient_data("cholesterol", top=3)
    print(f"Found {len(results)} results for 'cholesterol'")
    for result in results:
        print(f"- Patient: {result.get('patient_id')}, Date: {result.get('date')}")
        print(f"  Text: {result.get('text', '')[:100]}...")

    # Test 2: Get patient history
    print("\n2. Testing patient history retrieval...")
    history = rag.get_patient_history("moayad")
    print(f"Found {len(history)} records for patient 'moayad'")
    for record in history:
        print(f"- {record.get('date')}: {record.get('category')} - {record.get('text', '')[:80]}...")

    print("\n✅ RAG Manager tests completed!")
