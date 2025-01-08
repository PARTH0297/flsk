from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from groq import APIConnectionError
import logging
import os
import httpx

#httpx._config.verify = httpx.CertVerificationContext.from_file('C:\\Users\\NPAWAR8\\Downloads\\cacert.pem')
#httpx._config.verify_ssl = 'C:\\Users\\NPAWAR8\\Downloads\\cacert.pem'

def run_model(resume, job_description):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.environ['HTTPX_DISABLE_CERT_VERIFICATION'] = '1'
    
    try:
        chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

        system = """
        I want you to assess the compatibility between the given resume and job description, consider overall resume for comparison.
        Please provide a compatibility score in percentage.
        Additionally, provide accurate technical matching skills as well as technical missing skills,
        output in json format that too single word, I don't want explanation, I just need 3 keys -compatibility score,matching skills,missing skills
        """

        human = """
        Resume:
        {resume}

        Job Description:
        {job_description}
        """

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        chain = prompt | chat

        resp = chain.invoke({"resume": resume, "job_description": job_description})
        return resp
    except APIConnectionError as e:
        logging.error("Connection error: %s", e)
        return {"error": "Failed to connect to the Groq API. Please try again later."}