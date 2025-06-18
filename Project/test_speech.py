import requests
import os
from config import AZURE_SPEECH_KEY, AZURE_SPEECH_ENDPOINT

def test_speech_service():
    """Test Azure Speech service authentication"""

    print("=== Azure Speech Service Test ===")
    print(f"Endpoint: {AZURE_SPEECH_ENDPOINT}")
    print(f"API Key (first 10 chars): {AZURE_SPEECH_KEY[:10]}...")
    print(f"API Key length: {len(AZURE_SPEECH_KEY)} characters")

    # Test with a simple API call to check authentication
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/json"
    }

    # Try to get available voices (simple auth test)
    test_url = f"{AZURE_SPEECH_ENDPOINT}speechtotext/speech/voices?api-version=2024-11-15"

    try:
        response = requests.get(test_url, headers=headers)
        print(f"\nResponse Status: {response.status_code}")

        if response.status_code == 200:
            print("✅ Authentication successful!")
            print("Your Azure Speech service is working correctly.")
        elif response.status_code == 401:
            print("❌ Authentication failed!")
            print("Issue: Invalid API key or expired subscription")
            print("\nTroubleshooting steps:")
            print("1. Check your Azure Speech service in Azure Portal")
            print("2. Verify the API key is correct and not expired")
            print("3. Ensure the service is in the correct region")
        elif response.status_code == 403:
            print("❌ Permission denied!")
            print("Issue: API key doesn't have required permissions")
        else:
            print(f"❌ Unexpected error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Connection error: {str(e)}")

if __name__ == "__main__":
    test_speech_service()
