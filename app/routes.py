import json
from flask import Blueprint, request, jsonify
from groq import RateLimitError
from .langchain_integration import run_model
from .batch_process import batch_model
from .jd_parser import parse_job_description
import time
import logging

main = Blueprint('main', __name__)

# Configure logging to show only WARNING and ERROR messages
logging.basicConfig(level=logging.WARNING)

# Rate limit settings
TOKEN_LIMIT = 5000  # tokens per minute
tokens_used = 0
start_time = time.time()

def reset_token_count():
    global tokens_used, start_time
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:  # Reset every minute
        tokens_used = 0
        start_time = time.time()

def estimate_tokens(resume, job_description):
    # Simple estimation based on the length of the text
    return len(resume.split()) + len(job_description.split())

@main.route('/')
def home():
    return "Hello, Flask!"

@main.route('/process', methods=['POST'])
def process():
    data = request.json
    resume = data.get('resume')
    job_description = data.get('job_description')

    # Directly proceed with the model without input format checks or sanitization
    response = run_model(resume, job_description)
    return jsonify(response)



@main.route('/evaluate', methods=['POST'])
def evaluate():
    global tokens_used
    results = []
    data = request.json  # Expecting a list of structured job data

    if not isinstance(data, list):
        return jsonify({"error": "Input must be a list of objects with 'resume' and structured job details"}), 400

    for item in data:
        resume = item.get('resume')
        job_details = item.get('job_details')  # Structured job details

        # Validate if resume or job details are missing
        if not resume or not job_details:
            results.append({"error": "Missing resume or job details"})
            continue

        # Estimate token usage
        tokens_needed = estimate_tokens(resume, job_details)

        # Check if we can proceed without exceeding the limit
        reset_token_count()
        if tokens_used + tokens_needed > TOKEN_LIMIT:
            time_to_wait = (tokens_used + tokens_needed - TOKEN_LIMIT) / TOKEN_LIMIT * 60
            logging.warning(f"Rate limit exceeded. Waiting for {time_to_wait:.2f} seconds.")
            time.sleep(time_to_wait)

        # Process each resume using batch_model function
        try:
            response = batch_model(resume, job_details)
            tokens_used += tokens_needed  # Update token usage
        except RateLimitError as e:
            logging.error(f"Rate limit error: {e}. Retrying after {e.retry_after} seconds.")
            time.sleep(e.retry_after)
            response = batch_model(resume, job_details)  # Retry after waiting

        # Check for errors in response
        if 'error' in response:
            results.append({"error": response['error']})
        else:
            results.append(response)

    return jsonify(results)


@main.route('/parseJD', methods=['POST'])
def parse_jd():
    """
    API endpoint to parse Job Descriptions using AI.
    Expects a JSON with { "job_description": "JD text here" }.
    """
    data = request.json
    job_description = data.get('job_description')

    if not job_description:
        return jsonify({"error": "Missing job_description"}), 400

    try:
        response = parse_job_description(job_description)
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error parsing job description: {e}")
        return jsonify({"error": str(e)}), 500
