"""Client connector for Dynamic Multimodal Document AI Backend API."""

import os
import json
import requests

DEFAULT_BACKEND_URL = os.getenv(
    "DYNAMIC_AI_BACKEND_URL",
    os.getenv("STREAMLIT_API_URL", "https://dynamic-document-ai.onrender.com")
)


class DynamicExtractorClient:
    """Client to communicate with the dynamic multimodal document extraction API."""

    def __init__(self, base_url: str = DEFAULT_BACKEND_URL):
        self.base_url = str(base_url).strip().rstrip("/")

    def check_health(self) -> dict:
        """Probe health check endpoint.
        
        Returns minimal 204 check and does not trigger document extraction.
        """
        try:
            res = requests.get(f"{self.base_url}/health", timeout=8)
            if res.status_code in [200, 204]:
                return {
                    "online": True,
                    "status_code": res.status_code,
                    "details": {"status": "healthy", "code": res.status_code}
                }
            # Secondary check
            res_v1 = requests.get(f"{self.base_url}/api/v1/health", timeout=8)
            return {
                "online": res_v1.status_code in [200, 204],
                "status_code": res_v1.status_code,
                "details": res_v1.json() if res_v1.status_code == 200 else {}
            }
        except Exception as err:
            return {"online": False, "error": str(err)}

    def extract(self, filename: str, file_bytes: bytes, mime_type: str = "application/octet-stream",
                custom_instructions: str = None, schema_json: dict = None) -> dict:
        """Send document for multimodal extraction."""
        files = {"file": (filename, file_bytes, mime_type)}
        data = {}
        if custom_instructions:
            data["custom_instructions"] = custom_instructions
        if schema_json:
            data["schema_json"] = json.dumps(schema_json)

        endpoint = f"{self.base_url}/api/v1/extract/upload"
        response = requests.post(endpoint, files=files, data=data, timeout=180)
        if response.status_code == 200:
            return response.json()
        raise RuntimeError(f"Dynamic Extraction API failed (HTTP {response.status_code}): {response.text}")
