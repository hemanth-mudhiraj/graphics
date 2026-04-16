"""Gemini LLM integration for generating medical do's and don'ts."""

import os
import json
import requests


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def get_recommendations(disease: str, probability: float) -> dict:
    """
    Send the predicted disease to Gemini LLM and get do's and don'ts.
    Returns structured recommendations.
    """
    if not GEMINI_API_KEY:
        return _fallback_recommendations(disease)

    prompt = f"""You are a medical AI assistant. A chest X-ray analysis model has detected
"{disease}" with {probability*100:.1f}% confidence.

Provide medical guidance in the following JSON format (and ONLY valid JSON, no markdown):
{{
    "disease_name": "{disease}",
    "description": "Brief 2-3 sentence description of the condition",
    "severity": "low/moderate/high",
    "dos": [
        "5-6 specific things the patient SHOULD do"
    ],
    "donts": [
        "5-6 specific things the patient should NOT do"
    ],
    "when_to_see_doctor": "Brief guidance on when to seek immediate medical attention",
    "general_advice": "A brief reassuring note about the condition"
}}

Important: Provide practical, helpful advice. Include a disclaimer that this is AI-generated
and not a substitute for professional medical advice."""

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024,
                }
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Extract text from Gemini response
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Clean up: remove markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        recommendations = json.loads(text)
        recommendations["source"] = "gemini"
        return recommendations

    except Exception as e:
        print(f"Gemini API error: {e}")
        result = _fallback_recommendations(disease)
        result["error"] = str(e)
        return result


def _fallback_recommendations(disease: str) -> dict:
    """Provide basic fallback recommendations when Gemini API is unavailable."""
    return {
        "disease_name": disease,
        "description": f"{disease} has been detected in the chest X-ray analysis. Please consult a healthcare professional for accurate diagnosis and treatment.",
        "severity": "unknown",
        "dos": [
            "Consult a qualified healthcare professional immediately",
            "Bring this report to your doctor for review",
            "Follow your doctor's prescribed treatment plan",
            "Keep track of any symptoms you experience",
            "Maintain a healthy lifestyle with proper rest",
        ],
        "donts": [
            "Do not self-diagnose based solely on this AI analysis",
            "Do not ignore persistent symptoms",
            "Do not delay seeking medical attention",
            "Do not stop any current medications without consulting your doctor",
            "Do not rely on this report as a definitive diagnosis",
        ],
        "when_to_see_doctor": "Please see a doctor as soon as possible to discuss these findings.",
        "general_advice": "This is an AI-generated analysis and should not be used as a substitute for professional medical advice. Always consult a qualified healthcare provider.",
        "source": "fallback",
    }
