import logging
import json
import re
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq, APIConnectionError, RateLimitError


# List of API keys for rotation
API_KEYS = [
    "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",
    "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",
    "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",
    "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"
]
current_api_index = 0  # Index to track current API key usage


# Global variables for token management
tokens_used = 0
start_time = time.time()
TOKEN_LIMIT = 5000  # Tokens per minute

# Function to reset token count every minute
def reset_token_count():
    global tokens_used, start_time
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:  # Reset every minute
        tokens_used = 0
        start_time = time.time()


# Function to get the next API key in round-robin manner
def get_next_api_key():
    global current_api_index
    key = API_KEYS[current_api_index]
    current_api_index = (current_api_index + 1) % len(API_KEYS)  # Move to next key
    return key


def naive_json_from_text(text):
    # Regular expression to match everything between the first { and the last }
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))  # Try parsing the matched JSON
    except json.JSONDecodeError:
        return None  # Return None if JSON parsing fails


# Function to process a resume using Groq API with API key rotation and token management
def batch_model(resume, job_details, retry_attempts=3):
    global current_api_index, tokens_used, start_time

    reset_token_count()  # Reset token count if a minute has passed

    for attempt in range(len(API_KEYS) * retry_attempts):  # Try each API key multiple times before giving up
        api_key = get_next_api_key()
        try:
            client = Groq(api_key=api_key)

            system_prompt = """I want you to assess the compatibility between the given resume and job description and return a structured JSON object. Your response must strictly follow the format below:  

{{  
  "compatibilityPercentage": "The compatibility percentage between the resume and job description (numerical value between 0 and 100)",  
  "mustHaveSkills": [  
    {{  
      "skill": "Skill name (string) (same as mentioned in job description)",  
      "presentInResume": "Yes/No (string)",  
      "comment": "Details about the technical skill, its relevance to the job description, and experience level (e.g., 'Well-Demonstrated' or 'Missing' or 'Experienced (2 years)' or 'Used in Project X'). Mention experience details as well."  
    }}  
  ],  
  "goodToHaveSkills": [  
    {{  
      "skill": "Skill name (string) (same as mentioned in job description, if not provided in JD, use 'Null')",  
      "presentInResume": "Yes/No (string)",  
      "comment": "Details about the technical skill, its relevance to the job description, and experience level (e.g., 'Bonus Skill' or 'Missing')."  
    }}  
  ],  
  "extraSkills": [
  {{
    "skill": "Skill name (string) (technical skills from resume not explicitly mentioned in job description)",
    "presentInResume": "Yes (string)",
    "comment": "Details about the technical skill, its relevance to the job description, and experience level."
  }}
  ],
  "experience": {{  
    "overallIndustryExperience": "Total years of experience in the industry (integer value, e.g., 5). Ensure it is not a string.",  
    "companiesWorked": [  
      {{  
        "companyName": "Company name (string) or 'Confidential' if not mentioned",  
        "yearsWorked": "Number of years worked at the company (e.g., '2 years 5 months'). If the candidate is currently working at the company, use the format 'X years Y months (Present)'.",  
        "workDetails": "Brief description of work done in this company, including technologies used, role, key contributions, projects worked on, and responsibilities (e.g., 'Developed microservices for an e-commerce platform using Spring Boot and Kafka. Led API integrations and improved system performance by 30%')."  
      }}  
    ]  
  }}  
}}  


### Skill Classification:  
1. **Must-Have Skills**:  
   - **Ensure that all must-have skills mentioned in the job description are considered.**  
   - If any must-have skills are missing in the resume, mark them as "Missing" and explain their impact (e.g., "Missing. Impact: PostgreSQL is a critical must-have skill for this role. Penalty applied").
   - If a must-have skill is present in the resume, provide detailed comments about the candidate's experience with that skill.
   
2. **Good-to-Have Skills**:  
   - **Ensure that good-to-have skills in the job description are also checked.**  
   - If no good-to-have skills are mentioned in the job description, exclude the `goodToHaveSkills` key from the JSON response.
   
3. **Extra Skills**:  
   - Include technical skills found in the resume that were **not explicitly mentioned** in the job description.  
   - Provide 5 to 7 relevant extra skills from the resume and explain their relevance.
    

### Output Guidelines:  
- Focus only on technical skills** (exclude leadership, communication, problem-solving, etc.).  
- Ensure concise and meaningful comments for each skill, mentioning experience levels and context.
- Provide structured experience details**, including work done in each company.  
- Always provide a valid JSON response without additional text.  
"""
            human_prompt = f"Resume: {resume}\nJob Details:\n{job_details}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]

            # Invoke Groq API
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.1,
                top_p=1,
                stream=False,
                response_format={"type": "json_object"},
                stop=None,
            )

            # Extract response
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                response_content = completion.choices[0].message.content
            else:
                response_content = ""

            # Update token usage (estimate: input tokens + output tokens)
            tokens_used += len(response_content.split())  # Rough estimation using word count

            # Ensure token limit is not exceeded
            if tokens_used > TOKEN_LIMIT:
                logging.warning("Token limit reached. Waiting for reset...")
                time.sleep(60)  # Wait 1 minute for token reset
                reset_token_count()  # Reset token count
                continue  # Retry with the same API key after reset

            # Parse JSON response
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                logging.warning("Failed to parse JSON directly. Extracting JSON from text...")
                result_json = naive_json_from_text(response_content)
                if result_json is None:
                    raise ValueError("The Groq API response does not contain valid JSON.")
                return result_json

        except RateLimitError as e:
            # Extracting error details from Groq response
            error_message = e.message
            error_data = error_message.get('error', {})

            # Extract relevant rate limit error data
            tokens_requested = error_data.get('message', {}).get('requested', 0)
            retry_after = e.retry_after  # Time to wait before retrying

            # Check if requested tokens are less than the rate limit
            if tokens_requested < 5000:
                logging.error(f"Rate limit error: {error_message}. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)  # Sleep for the specified retry time
                return batch_model(resume, job_details)  # Retry the request
            else:
                logging.warning("Rate limit exceeded with more than 5000 tokens requested, skipping this request.")
                return {"error": "Rate limit exceeded, skipping this request."}

        except APIConnectionError as e:
            logging.error(f"Connection error with API key {api_key}: {e}")
            continue  # Try the next API key

        except Exception as e:
            logging.exception(f"Unexpected error with API key {api_key}: {e}")
            continue  # Try the next API key

    logging.error("All API keys have reached their rate limits. Please try again later.")
    return {"error": "All API keys exhausted. Try again later."}