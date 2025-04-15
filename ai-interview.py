import os
import json
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import tempfile
import soundfile as sf
from kokoro import KPipeline
# import pyaudio
import wave
import whisper
import numpy as np
import time
import threading
from st_audiorec import st_audiorec
import base64
import subprocess
from streamlit_ace import st_ace #type:ignore

# Audio Configuration
# FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
MAX_RECORD_TIME = 120  # Increased recording time
SILENCE_THRESHOLD = 500
SILENCE_TIMEOUT = 5
TEMP_FILENAME = "response.wav"

load_dotenv()

api_key=os.environ.get("GROQ_API_KEY")

client = Groq() 

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'job_description_input' not in st.session_state:
    st.session_state.job_description_input = ""
if "questions_locked" not in st.session_state:
    st.session_state.questions_locked = False
if "last_locked_jd" not in st.session_state:
    st.session_state.last_locked_jd = ""
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None

    
def generate_audio(text, voice='af_heart', speed=1.0):
    """Generate and return audio file path from text"""
    tts_pipeline = KPipeline(lang_code='a')
    generator = tts_pipeline(text, voice=voice, speed=speed)
    audio_data = b''
    for i, (_, _, audio) in enumerate(generator):
        audio_data = audio  # Only save last or override with all audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, 24000)
    return temp_file.name

def generate_interview_questions(job_description, model="llama3-70b-8192"):
    system_prompt = """You are a technical hiring manager expert. Generate 5 interview questions (mix of technical concepts and coding problems) based on the job description. 
Follow these guidelines:
2. Ensure **no repetition of concepts, topics, or patterns** across the questions.
3. For conceptual questions:
   - Cover different areas of Job role 
   - Provide a concise sample answer
4. For coding questions:
   - Make them **language-agnostic** (suitable for any programming language)
   - Focus on algorithmic thinking and logic, **not** on syntax
   - Provide **no solutions**
   - Include clear input/output examples 

# Prioritize fundamental problems that test programming logic: 
# - Array/string manipulation 
# - Basic data structures 
# - Simple algorithms 
# - Problem decomposition 
# - Also u can genrate question based on Job role or Job description

Important:
- Avoid repeating the same types of problems or rephrasing similar ones.



Return format: must be JSON array with objects containing: 
- For conceptual questions: {"type": "concept", "question": "", "answer": ""} 
- For coding questions: {"type": "coding", "question": "", "input_example": "", "output_example": ""}

Your goal: generate **diverse, job-relevant**, and well-scoped questions.
"""

# system_prompt = """You are a technical hiring manager expert. Generate 5 interview questions (mix of technical concepts and coding problems) based on the job description. 
# Follow these guidelines: 
# 1. For conceptual questions: provide a sample answer 
# 2. For coding questions: - Make them language-agnostic (can be implemented in any programming language)
#  - Focus on logic/algorithms rather than language-specific features 
#  - Do NOT provide solutions (since we don't know the candidate's preferred language) 
#  - Include clear input/output examples 
# Return format: must be JSON array with objects containing: 
# - For conceptual questions: {"type": "concept", "question": "", "answer": ""} 
# - For coding questions: {"type": "coding", "question": "", "input_example": "", "output_example": ""}
# Prioritize fundamental problems that test programming logic: 
# - Array/string manipulation 
# - Basic data structures 
# - Simple algorithms 
# - Problem decomposition 
# - Also u can genrate question based on Job role or Job description """


    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": job_description}
            ],
            model=model,
            response_format={"type": "json_object"},
            temperature=0.5,
        )

        content = json.loads(response.choices[0].message.content)
        return content.get("questions", [])
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def save_questions(questions):
    with open('interview_data.json', 'w') as f:
        json.dump({"questions": questions}, f)

def load_questions():
    try:
        with open('interview_data.json', 'r') as f:
            data = json.load(f)
            return data.get("questions", [])
    except FileNotFoundError:
        return []
    
def evaluate_answer_with_llm(user_answer, sample_answer, model="llama3-70b-8192"):
    """Evaluate user answer against sample using AI semantic analysis"""
    system_prompt = """You are an expert technical interviewer. Evaluate user responses differently based on question type:

For CONCEPTUAL QUESTIONS (type="concept"):
1. Technical accuracy
2. Key concepts covered
3. Depth of understanding
4. Relevance to the question
** Focus on semantic similarity rather than exact wording **

For CODING QUESTIONS (type="coding"):
1. Functional correctness (test against 3-5 edge cases)
2. Algorithmic efficiency
3. Code readability/structure
4. Explanation clarity (if provided)
5. Language-agnostic best practices

EVALUATION RULES:
- Never use phrases like "perfect match" or "identical to sample"
- For coding: Generate test cases based on the problem statement
- Adapt feedback to the user's implementation language
- Your feedback should include any grammar mistakes in concept tupe question with corrections, and point out technical deficiencies based on question along with suggestions for improvement. 
- Provide detailed feedback of candidate answer (no limits for length)  
- Penalize hardcoded solutions passing only given examples

**Don't cut the score for grammar mistake**

Return JSON format: {
    "score": 0-10 (10=excellent),
    "feedback": "constructive feedback",
    "test_cases": [{"input": "", "output": "", "passed": bool}] (coding only),
    "language": "detected_programming_language" (coding only)
}

EXAMPLE CODING EVALUATION:
Input Question: "Write a function that checks if a string is a palindrome"
User Code: "def is_pal(s): return s == s[::-1]"
Response:
{
    "score": 9,
    "feedback": "Solution correctly implements palindrome check but could handle case sensitivity. Consider edge cases like empty strings.",
    "test_cases": [
        {"input": "'racecar'", "output": "True", "passed": true},
        {"input": "'Racecar'", "output": "False", "passed": true},
        {"input": "''", "output": "True", "passed": true}
    ],
    "language": "Python"
}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SAMPLE ANSWER: {sample_answer}\nUSER ANSWER: {user_answer}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return {"score": 0, "feedback": "Evaluation failed"}
    
def evaluate_answers():
    try:
        with open('interview_data.json', 'r') as f:
            data = json.load(f)
            questions = data.get("questions", [])
            
            total_points = 0
            max_points = len(questions) * 10
            
            for q in questions:
                user_answer = q.get('user_answer', '')
                sample_answer = q.get('answer', '')
                
                if not user_answer:
                    q.update({"points": 0, "feedback": "No answer provided"})
                    continue
                
                # AI-powered evaluation
                evaluation = evaluate_answer_with_llm(user_answer, sample_answer)
                q['points'] = min(10, max(0, int(evaluation.get('score', 0))))
                q['feedback'] = evaluation.get('feedback', 'No feedback generated')
                total_points += q['points']

            data['result'] = {
                'total_points': total_points,
                'percentage': (total_points / max_points) * 100 if max_points > 0 else 0,
                'status': 'Pass' if (total_points / max_points) * 100 >= 70 else 'Fail'
            }
            
            with open('interview_data.json', 'w') as f:
                json.dump(data, f)
            
            return data['result']
    except Exception as e:  
        st.error(f"Evaluation error: {str(e)}")
        return None

# Page 1: Generate Questions
def page_generate_questions():
    st.header("Generate Interview Questions")

    job_desc = st.text_area("Paste Job Description Here",
                            value=st.session_state.job_description_input,
                            height=200,
                            key="job_desc_input")

    jd_changed = job_desc.strip() != st.session_state.job_description_input.strip()

    # If JD has changed, unlock
    if jd_changed:
        st.session_state.questions_locked = False
        st.session_state.generated = False
        st.session_state.last_locked_jd = ""
        if "questions" in st.session_state:
            del st.session_state.questions
        open('interview_data.json', 'w').close()

    col1, col2 = st.columns([1, 3])

    # Buttons — disable if locked
    generate_btn = col1.button("Generate Questions",
                               disabled=st.session_state.generated or st.session_state.questions_locked)

    regenerate_btn = False
    lock_btn = False

    if st.session_state.generated:
        with col2:
            col2a, col2b = st.columns([1, 1])
            if st.session_state.questions_locked:
                pass
            else:
                regenerate_btn = col2a.button("Regenerate Questions", disabled=st.session_state.questions_locked)
                lock_btn = col2b.button("Lock Questions", disabled=st.session_state.questions_locked)

    # Generate or Regenerate
    if (generate_btn or regenerate_btn) and not st.session_state.questions_locked:
        if job_desc.strip():
            st.session_state.job_description_input = job_desc
            
            if regenerate_btn:
                if "questions" in st.session_state:
                    del st.session_state.questions
                open('interview_data.json', 'w').close()
                st.session_state.generated = False

            questions = generate_interview_questions(job_desc)
            if questions:
                save_questions(questions)
                st.session_state.generated = True
                st.session_state.questions = questions
                st.success("Questions generated successfully!" if generate_btn else "Questions regenerated successfully!")
                st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
            else:
                st.error("Failed to generate questions")
        else:
            st.warning("Please enter a job description")

    # Locking
    if lock_btn:
        with st.spinner("Locking questions and generating audio..."):
            st.session_state.questions_locked = True
            st.session_state.last_locked_jd = job_desc.strip()

            # Generate audio for each question
            audio_dir = "question_audios"
            os.makedirs(audio_dir, exist_ok=True)
            questions = st.session_state.get("questions", load_questions())
            for i, q in enumerate(questions):
                audio_path = os.path.join(audio_dir, f"q{i+1}.wav")
                audio_path_generated = generate_audio(q["question"])
                os.replace(audio_path_generated, audio_path)

            st.success("Questions locked. Change the job description to unlock and regenerate.")


        st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
        
    # Show questions
    if st.session_state.generated:
        st.subheader("Generated Questions")
        questions = st.session_state.get("questions", load_questions())

        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}:** {q['question']}")
            st.divider()

def execute_code(language, code):
    result = {"output": "", "error": ""}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            if language == "python":
                with open(os.path.join(temp_dir, "code.py"), "w") as f:
                    f.write(code)
                process = subprocess.run(
                    ["python3", os.path.join(temp_dir, "code.py")],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif language == "c":
                with open(os.path.join(temp_dir, "code.c"), "w") as f:
                    f.write(code)
                compile_process = subprocess.run(
                    ["gcc", os.path.join(temp_dir, "code.c"), "-o", os.path.join(temp_dir, "out")],
                    capture_output=True,
                    text=True
                )
                if compile_process.returncode != 0:
                    result["error"] = compile_process.stderr
                    return result
                process = subprocess.run(
                    [os.path.join(temp_dir, "out")],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            elif language == "java":
                with open(os.path.join(temp_dir, "Main.java"), "w") as f:
                    f.write(code)
                compile_process = subprocess.run(
                    ["javac", os.path.join(temp_dir, "Main.java")],
                    capture_output=True,
                    text=True
                )
                if compile_process.returncode != 0:
                    result["error"] = compile_process.stderr
                    return result
                process = subprocess.run(
                    ["java", "-cp", temp_dir, "Main"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            result["output"] = process.stdout
            result["error"] = process.stderr

    except subprocess.TimeoutExpired:
        result["error"] = "Execution timed out"
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Page 2: Answer The Questions
def page_answer_questions():
    st.header("Answer Interview Questions")
    questions = load_questions()
    
    if not questions:
        st.warning("No questions found. Generate questions first!")
        return

    if len(st.session_state.answers) != len(questions):
        st.session_state.answers = [""] * len(questions)

    idx = st.session_state.current_question
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if idx > 0:
            if st.button("Previous"):
                st.session_state.current_question -= 1
                st.rerun()
    with col3:
        if idx < len(questions) - 1:
            if st.button("Next"):
                st.session_state.current_question += 1
                st.rerun()

    # Question Display
    st.subheader(f"Question {idx + 1}/{len(questions)}")
    current_q = questions[idx]
    st.markdown(f"**{current_q['question']}**")

    # Audio of current question
    audio_path = os.path.join("question_audios", f"q{idx+1}.wav")
    if os.path.exists(audio_path):
        st.audio(audio_path, format='audio/wav',autoplay=True)
    else:
        st.warning("Audio missing for this question")
    
    # Show input/output examples if coding question
    if current_q.get('type') == 'coding':
     
        if 'input_example' in current_q:
            st.markdown(f"*Input Example:* `{current_q['input_example']}`")
        if 'output_example' in current_q:
            st.markdown(f"*Expected Output:* `{current_q['output_example']}`")
        
        # Empty code block (you'll paste your implementation here)
        # Streamlit UI
        st.title("Code Editor")

        # Language selection
        language = st.selectbox("Select Language", ["python", "c", "java"])
        theme = st.selectbox("Select Theme", ["chrome", "twilight", "github"])
        font_size = st.slider("Font Size", 10, 24, 18)

        # Code editor
        code = st_ace(
            placeholder="Write your code here...",
            language=language,
            theme=theme,
            key=f"editor{language}",
            font_size=font_size,
            min_lines=30,
            keybinding="vscode",
            
        )

        if st.button("Run Code"):
            if code.strip() == "":
                st.error("Please write some code first")
            else:
                result = execute_code(language, code)
                
                if result["error"]:
                    st.error("Execution Error:")
                    st.code(result["error"], language="bash")
                else:
                    st.success("Output:")
                    st.code(result["output"], language="bash")

        if st.button("Submit Code"):
            st.subheader("Submitted Code")
            st.code(code, language=language) # Empty code block
            st.session_state.answers[idx] = code if code.strip() != None else "NA"
    
    # Audio playback (for conceptual questions only)
    if current_q.get('type') != 'coding':
    
        # Voice Answer Section for conceptual questions
        st.markdown("### Your Answer")
        
        # Recording and transcription
        result = st_audiorec()
        
        if result is not None:
            file_path, transcription = result
            if transcription:
                st.session_state.answers[idx] = transcription
                st.success("Transcription received!")
        else:
            transcription = st.session_state.answers[idx]  # fallback to existing if available
        
        # Text area with current transcription
        answer_text = st.text_area(
            "Answer will appear here",
            transcription,
        )
        if answer_text:
            st.session_state.answers[idx] = answer_text 
        else:
            st.session_state.answers[idx] = "NA"

    if st.button("Submit All Answers"):
        for i in range(len(questions)):
            questions[i]['user_answer'] = st.session_state.answers[i]
        
        with open('interview_data.json', 'w') as f:
            json.dump({"questions": questions}, f)
        
        st.success("Answers submitted successfully!")

# Page 3: Results display to show feedback
def page_view_results():
    st.header("Interview Results")
    
    if st.button("Evaluate Answers"):
        result = evaluate_answers()
        if result:
            st.subheader("Final Result")
            st.markdown(f"""
            **Total Points:** {result['total_points']}  
            **Percentage:** {result['percentage']:.2f}%  
            **Status:** {result['status']}
            """)
            
            st.subheader("Detailed Evaluation")
            questions = load_questions()
            for i, q in enumerate(questions, 1):
                st.markdown(f"**Q{i}:** {q['question']}")
                st.markdown(f"**Your Answer:** {q.get('user_answer', 'No answer')}")
                st.markdown(f"**Score:** {q.get('points', 0)}/10")
                st.markdown(f"**Feedback:** {q.get('feedback', 'No feedback available')}")
                st.divider()

# Main App
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
        ["Generate Questions", "Answer Questions", "View Results"])
    
    if page == "Generate Questions":
        page_generate_questions()
    elif page == "Answer Questions":
        page_answer_questions() 
    elif page == "View Results":
        page_view_results()

if __name__ == "__main__":
    main()