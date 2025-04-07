import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_interview_questions(job_description, model="llama3-70b-8192"):
    system_prompt = """You are a hiring manager expert. Generate 10 technical interview questions 
    based on the job description. For each question, provide a sample answer. 
    Return format: JSON array with {question, answer} objects."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": job_description}
        ],
        model=model,
        response_format={"type": "json_object"}
    )

    try:
        questions = json.loads(response.choices[0].message.content)["questions"]
        save_to_file(questions)
        return True
    except:
        return []

def save_to_file(questions, filename="interview_questions.json"):
    with open(filename, 'w') as f:
        json.dump(questions, f, indent=2)

def load_from_file(filename="interview_questions.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    
generate_interview_questions("python developer 3+ year expirence")