import logging
import os
import json
import httpx
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from groq import APIConnectionError


def naive_json_from_text(text):
    # Regular expression to match everything between the first { and the last }
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))  # Try parsing the matched JSON
    except json.JSONDecodeError:
        return None  # Return None if JSON parsing fails





def batch_model(resume, job_description):
    # Configure logging
    #logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set up SSL/TLS handling securely
    # cacert_path = 'C:\\Users\\NPAWAR8\\Downloads\\cacert.pem'  # Update to your actual path
    # if os.path.exists(cacert_path):
    #     httpx._config.verify = cacert_path
    # else:
    #     logging.warning("SSL certificate not found, disabling verification.")
    #     os.environ['HTTPX_DISABLE_CERT_VERIFICATION'] = '1'

    try:
        # Initialize Groq Chat Model
        chat = ChatGroq(
            temperature=0,
            groq_api_key="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",
            model_name="mixtral-8x7b-32768"
        )

        # System and Human Prompt Definitions
        system_prompt = """I want you to assess the compatibility between the given resume and job description. Your output must strictly be a JSON object that maps to the following structure:
{{
  "compatibilityPercentage": "The compatibility percentage between the resume and job description (numerical value between 0 and 100)",
  "mustHaveSkills": [
    {{
      "skill": "Skill name (string)",
      "presentInResume": "Yes/No (string)",
      "comment": "Details about the technical skill, its relevance to the job description, and experience level (e.g., 'Well-Demonstrated' or 'Missing' or 'Experienced (2 years)' or 'Used in Project X')"
    }}
  ],
  "goodToHaveSkills": [
    {{
      "skill": "Skill name (string)",
      "presentInResume": "Yes/No (string)",
      "comment": "Details about the technical skill, its relevance to the job description, and experience level (e.g., 'Bonus Skill' or 'Missing')"
    }}
  ],
  "extraSkills": [
    {{
      "skill": "Skill name (string)",
      "presentInResume": "Yes (string)",
      "comment": "Details about the technical skill, its relevance to the job description, and experience level (e.g., 'Not mentioned in JD')"
    }}
  ]
}}

Your response must strictly follow this format. Classify skills into these categories:
1. **Must-Have Skills**: Critical technical skills present in both the resume and job description.
2. **Good-to-Have Skills**: Technical skills that are desirable but not critical, and present in both.
3. **Extra Skills**: Technical skills found in the resume but not listed in the job description(only give 7 to 20 Extra Skills major ones).

Please focus on **technical skills** only, excluding any **soft skills** such as leadership, communication, and problem-solving abilities. For each skill, include a description of the level of experience and relevance to the job role.
"""




        human_prompt = """Resume: {resume}\nJob Description: {job_description}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        # Chain execution
        chain = prompt | chat

        # Invoke Chat Model
        logging.info("Sending request to Groq model...")
        resp = chain.invoke({"resume": resume, "job_description": job_description})

        # # Parse JSON response
        # try:
        #     result_json = json.loads(resp.content)
        #     return result_json
        # except json.JSONDecodeError:
        #     logging.error("Failed to parse JSON from response.")
        #     raise ValueError("The Groq API returned an invalid JSON response.")

                # Try parsing the response as JSON directly
        try:
            result_json = json.loads(resp.content)
            return result_json
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON directly. Attempting to extract JSON from the text...")
            # Use naive approach to extract potential JSON from the response text
            result_json = naive_json_from_text(resp.content)
            if result_json is None:
                raise ValueError("The Groq API response does not contain valid JSON.")
            return result_json



    except APIConnectionError as e:
        logging.error("Groq API connection error: %s", e)
        return {"error": "Failed to connect to the Groq API. Please try again later."}
    except Exception as e:
        logging.exception("Unexpected error occurred.")
        return {"error": str(e)}

