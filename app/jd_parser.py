import logging
import json
import re
from langchain_groq import ChatGroq
from groq import Groq, APIConnectionError

def naive_json_from_text(text):
    """Extract JSON from text using regex."""
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def parse_job_description(job_description):
    """
    Uses Groq AI to extract structured details from a Job Description.
    Returns a JSON object with parsed data.
    """
    try:
        api_key = "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"
        client = Groq(api_key=api_key)

        # System and Human Prompt Definitions
        system_prompt = """Extract structured details from the given job description and return a JSON object.
        
        **Output Format:**
        {{
            "role": "Job title or role",
            "companyName": "Company name if mentioned, else empty (extract name if in website link or any other give it)",
            "location": "Job location if mentioned, else empty",
            "experienceMin": "Minimum experience in years (type int, if mentioned, else empty)",
            "experienceMax": "Maximum experience in years (type int, if mentioned, else empty)",
            "numOpenings": "Number of openings if mentioned, else empty"(type int),
            "mustHaveSkills": ["List of must-have skills,in response formatted concisely"],
            "goodToHaveSkills": ["List of good-to-have skills,in response formatted concisely"],
            "educationRequirement": "Required education qualification if mentioned, else empty",
            "certifications": ["List of certifications if mentioned, else empty"],
            "responsibilities": ["List of responsibilities"],
            "softSkills": ["List of soft skills if mentioned, else empty"]
        }}

        - Extract only **technical** skills under `mustHaveSkills` and `goodToHaveSkills`.
        - **Soft skills** (like teamwork, leadership, communication) should be listed under `softSkills`.
        - **Responsibilities** should be action-driven statements found in the JD.
        - If certain fields (like `companyName`, `experienceMin`, etc.) are missing, return empty strings.
        - **Ensure valid JSON output only.**
        """

        human_prompt = f"Job Description: {job_description}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ]

        # Invoke Groq Client
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.1,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )

        # Extract response content
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            response_content = completion.choices[0].message.content
        else:
            response_content = ""

        # Parse JSON response
        try:
            result_json = json.loads(response_content)
            return result_json
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON directly. Attempting regex extraction...")
            result_json = naive_json_from_text(response_content)
            if result_json is None:
                raise ValueError("The Groq API response does not contain valid JSON.")
            return result_json

    except APIConnectionError as e:
        logging.error(f"Groq API connection error: {e}")
        return {"error": "Failed to connect to the Groq API. Please try again later."}
    except Exception as e:
        logging.exception("Unexpected error occurred.", e)
        return {"error": str(e)}
