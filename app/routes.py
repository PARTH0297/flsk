from flask import Blueprint, request, jsonify
from groq import RateLimitError
from .langchain_integration import run_model
from .batch_process import batch_model
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

    response = run_model(resume, job_description)
    
    return jsonify(response)

@main.route('/evaluate', methods=['POST'])
def evaluate():
    global tokens_used
    results = []

    # Expecting a list of {resume, job_description} pairs
    data = request.json

    if not isinstance(data, list):
        return jsonify({"error": "Input must be a list of {resume, job_description} objects"}), 400

    for item in data:
        resume = item.get('resume')
        job_description = item.get('job_description')

        # Validate inputs
        if not resume or not job_description:
            results.append({"error": "Missing resume or job_description for one of the entries"})
            continue

        # Estimate token usage
        tokens_needed = estimate_tokens(resume, job_description)
        
        # Check if we can proceed without exceeding the limit
        reset_token_count()
        if tokens_used + tokens_needed > TOKEN_LIMIT:
            time_to_wait = (tokens_used + tokens_needed - TOKEN_LIMIT) / TOKEN_LIMIT * 60  # Calculate wait time
            logging.warning(f"Rate limit exceeded. Waiting for {time_to_wait:.2f} seconds.")  # Changed from INFO to WARNING
            time.sleep(time_to_wait)
        
        # Process each pair using the batch_model function
        try:
            response = batch_model(resume, job_description)
            tokens_used += tokens_needed  # Update token usage
        except RateLimitError as e:
            logging.error(f"Rate limit error: {e}. Retrying after {e.retry_after} seconds.")
            time.sleep(e.retry_after)
            response = batch_model(resume, job_description)  # Retry after waiting

        # Check for errors in the response
        if 'error' in response:
            results.append({"error": response['error']})
        else:
            results.append(response)

    # Return the batch of results as a JSON response
    return jsonify(results)
