import os
import json
import uuid
import tempfile
import threading
import subprocess
import shutil
import whisper



import streamlit as st
import soundfile as sf
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from kokoro import KPipeline
from st_audiorec import st_audiorec
from streamlit_ace import st_ace  # type:ignore

# Audio Configuration
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

# Generate and store a unique UUID for each user session if not already present
if "generated" not in st.session_state:
    st.session_state.generated = False
if "questions_locked" not in st.session_state:
    st.session_state.questions_locked = False
if "awaiting_username" not in st.session_state:
    st.session_state.awaiting_username = False
if 'user_uuid' not in st.session_state:
    st.session_state.user_uuid = str(uuid.uuid4())

# Ensure required directories exist
TEMP_QUESTIONS_FILE = "temp_questions.json"
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)
# Audio files will be stored in a user-specific directory (created as needed)

# Helper functions for per-user file handling
def get_data_filename():
    # Each user gets his/her own JSON file
    return os.path.join(USER_DATA_DIR, f"interview_data_{st.session_state.user_uuid}.json")

def save_questions(questions, username):
    user_dir = os.path.join(USER_DATA_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    filepath = os.path.join(user_dir, "questions.json")
    with open(filepath, 'w') as f:
        json.dump({"questions": questions}, f)

def save_temp_questions(questions):
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    filepath = os.path.join(USER_DATA_DIR, TEMP_QUESTIONS_FILE)
    with open(filepath, "w") as f:
        json.dump({"questions": questions}, f)


def load_questions(username):
    user_dir = os.path.join(USER_DATA_DIR, username)
    filename = os.path.join(user_dir, "questions.json")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get("questions", [])
    except FileNotFoundError:
        return []
# Audio generation function remains similar; file naming is handled at lock time
def generate_audio(text, voice='af_heart', speed=1.0):
    """Generate and return audio file path from text"""
    tts_pipeline = KPipeline(lang_code='a')
    generator = tts_pipeline(text, voice=voice, speed=speed)
    audio_data = b''
    for i, (_, _, audio) in enumerate(generator):
        audio_data = audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, 24000)
    return temp_file.name

def generate_interview_questions(job_description, model="llama3-70b-8192"):
    system_prompt = """You are a technical hiring manager expert. Generate 5 interview questions (mix of technical concepts and coding problems) based on the job description. 
Follow these guidelines:
1. Ensure **no repetition of concepts, topics, or patterns** across the questions.
2. For conceptual questions:
   - Cover different areas of Job role 
   - Provide a concise sample answer
3. For coding questions:
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


def evaluate_answer_with_llm(user_answer, sample_answer, question_type, model="llama3-70b-8192"):
    """Evaluate user answer against sample using AI semantic analysis"""
    
    system_prompt = """You are an expert technical interviewer. Evaluate user responses with balanced rigor, focusing on core understanding and functionality:

    For CONCEPTUAL QUESTIONS (type="concept"):
    1. Technical accuracy (primary factor)
    2. Key concepts coverage
    3. Practical understanding
    4. Logical coherence
    ** Accept equivalent technical terms and reasonable paraphrasing **
    ** Only note grammar issues if they affect understanding **

    For CODING QUESTIONS (type="coding"):
    1. Functional correctness (test through 3-5 logical cases)
    2. Algorithm appropriateness
    3. Basic code structure
    4. Solution explanation (if provided)
    5. Fundamental best practices

    EVALUATION PRINCIPLES:
    - Focus on substance over form - solutions should work first
    - Generate relevant test cases, not exhaustive ones
    - Documentations/comments are nice-to-have, not mandatory
    - Deduct max 0.5 points for missing non-critical elements
    - Acknowledge multiple valid approaches
    - Give benefit of doubt for minor style issues

    Return scores using this scale:
    9-10: Fully functional/accurate with minor improvements possible
    7-8: Working solution with non-critical gaps
    5-6: Partial solution with some valid elements
    Below 5: Fundamentally incorrect approach

    Return a valid JSON object for coding question only. Format:
    {
        "score": 0-10,
        "feedback": "Balanced feedback with core strengths, test results, and constructive suggestions",
        "passed_cases": 3,
        "total_cases": 5,
    }

    Return a valid JSON object for concept question only. Format:
    {
        "score": 0-10,
        "feedback": "Focus on knowledge gaps, not phrasing",
    }
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUESTION TYPE: {question_type.upper()}\nSAMPLE ANSWER: {sample_answer}\nUSER ANSWER: {user_answer}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        result = json.loads(response.choices[0].message.content)

        # Remove test case fields for concept questions if returned mistakenly
        if question_type == "concept":
            result.pop("passed_cases", None)
            result.pop("total_cases", None)
            result.pop("language", None)

        return result

    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return {"score": 0, "feedback": "Evaluation failed"}

def evaluate_answers(username):
    user_dir = os.path.join(USER_DATA_DIR, username)
    filename = os.path.join(user_dir, "questions.json")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            questions = data.get("questions", [])
            
            total_points = 0
            max_points = len(questions) * 10
            
            for q in questions:
                user_answer = q.get('user_answer', '')
                sample_answer = q.get('answer', '')
                question_type = q.get('type', '')
                if not user_answer:
                    q.update({"points": 0, "feedback": "No answer provided"})
                    continue
                
                evaluation = evaluate_answer_with_llm(user_answer, sample_answer,question_type)
                q['points'] = min(10, max(0, int(evaluation.get('score', 0))))
                q['feedback'] = evaluation.get('feedback', 'No feedback generated')
                
                # Save test case result summary if available
                if 'passed_cases' in evaluation and 'total_cases' in evaluation:
                    q['passed_cases'] = evaluation['passed_cases']
                    q['total_cases'] = evaluation['total_cases']
                
                total_points += q['points']

            data['result'] = {
                'total_points': total_points,
                'percentage': (total_points / max_points) * 100 if max_points > 0 else 0,
                'status': 'Pass' if (total_points / max_points) * 100 >= 70 else 'Fail'
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data['result']
    except Exception as e:  
        st.error(f"Evaluation error: {str(e)}")
        return None

# Page 1: Generate Questions
def page_generate_questions():
    st.header("Generate Interview Questions")
    
    # Initialize session state variables if they don't exist
    if "temp_questions" not in st.session_state:
        st.session_state.temp_questions = []
    if "questions_locked" not in st.session_state:
        st.session_state.questions_locked = False
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if "awaiting_username" not in st.session_state:
        st.session_state.awaiting_username = False
    
    # Job description input
    job_desc = st.text_area("Paste Job Description Here", height=200, key="job_desc_input")
    
    # Check if job description has changed
    jd_changed = job_desc.strip() != st.session_state.get("job_description_input", "").strip()
    
    if jd_changed:
        st.session_state.job_description_input = job_desc.strip()
        st.session_state.questions_locked = False
        st.session_state.generated = False
        st.session_state.last_locked_jd = ""
        if "questions" in st.session_state:
            del st.session_state.questions
    
    # Button layout
    col1, col2 = st.columns([1, 3])
    
    # Generate button - disable only if questions are locked
    generate_btn = col1.button("Generate Questions", disabled=st.session_state.get("questions_locked", False))
    
    # Default values for other buttons
    regenerate_btn = False
    lock_btn = False
    temp_file = os.path.join(USER_DATA_DIR, "temp_questions.json")
    # Show regenerate and lock buttons if questions were generated
    if st.session_state.get("generated", False):
        with col2:
            col2a, col2b = st.columns([1, 1])
            if not st.session_state.get("questions_locked", False):
                regenerate_btn = col2a.button("Regenerate Questions")
                lock_btn = col2b.button("Lock Questions")
    
    # Handle question generation
    if (generate_btn or regenerate_btn) and not st.session_state.get("questions_locked", False):
        if job_desc.strip():

            if regenerate_btn:
                if "questions" in st.session_state:
                    del st.session_state.questions
                open(temp_file, 'w').close()
                # st.session_state.generated = False

            # Generate questions
            questions = generate_interview_questions(job_desc)
            
            if questions:
                # Save questions to session state and temporary file
                st.session_state.questions = questions
                st.session_state.temp_questions = questions
                st.session_state.generated = True
                
                # Save to temporary file
                temp_file = os.path.join(USER_DATA_DIR, "temp_questions.json")
                with open(temp_file, "w") as f:
                    json.dump({"questions": questions}, f)
                
                st.success("Questions generated successfully!")
                st.rerun()
        else:
            st.warning("Please enter a job description")
    
    # Handle lock button and username input
    if lock_btn:
        st.session_state.awaiting_username = True
    
    # Show username input if awaiting username
    if st.session_state.get("awaiting_username", False):
        username = st.text_input("Enter a username to lock questions:")
        submit_btn = st.button("Submit Username")
        
        if submit_btn and username:
            # Create user directory
            user_dir = os.path.join(USER_DATA_DIR, username)
            os.makedirs(user_dir, exist_ok=True)
            
            # Save questions with username
            questions_file = os.path.join(user_dir, "questions.json")
            with open(questions_file, "w") as f:
                json.dump({"questions":st.session_state.questions}, f)
            
            # Generate audio files
            audio_dir = os.path.join(user_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            for i, q in enumerate(st.session_state.questions):
                audio_path = os.path.join(audio_dir, f"q{i+1}.wav")
                audio_path_generated = generate_audio(q["question"])
                os.replace(audio_path_generated, audio_path)
            
            # Update session state
            st.session_state.questions_locked = True
            st.session_state.awaiting_username = False
            st.session_state.username = username
            
            
            st.success(f"Questions locked for {username}!")
            st.rerun()
    
    # Display generated questions
    if st.session_state.get("generated", False):
        st.subheader("Generated Questions")
        
        # Try to get questions from session state, or load from temp file if they exist
        questions = st.session_state.get("questions", [])
        if not questions and os.path.exists(os.path.join(USER_DATA_DIR, "temp_questions.json")):
            try:
                with open(os.path.join(USER_DATA_DIR, "temp_questions.json"), "r") as f:
                    questions = json.load(f)
                st.session_state.questions = questions
            except:
                pass
        
        # Display questions
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

def generate_text_from_audio(bytes):
    if bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(bytes)
            temp_audio_path = tmp_file.name  # Store the temporary file path

        # Use Whisper model to transcribe the audio
        model = whisper.load_model("base")  # You can use other model sizes like "tiny", "small", "large", etc.
        result = model.transcribe(temp_audio_path)
        transcribed_text = result["text"]

        return temp_audio_path, transcribed_text  # Return audio file path and transcribed text
    return None, None


# Page 2: Answer The Questions
def page_answer_questions():
    st.header("Answer Interview Questions")
    
    if 'current_username' not in st.session_state:
        username = st.text_input("Enter your username:")
        if st.button("Load Questions"):
            user_dir = os.path.join(USER_DATA_DIR, username)
            
            if os.path.exists(os.path.join(user_dir, "questions.json")):
                st.session_state.current_username = username
                st.rerun()
            else:
                st.error("Invalid username")
        return
    
    username = st.session_state.current_username
    user_dir = os.path.join(USER_DATA_DIR, username)
    questions = load_questions(username)
    
    
    if not questions:
        st.warning("No questions found. Generate questions first!")
        return

    if len(st.session_state.get("answers", [])) != len(questions):
        st.session_state.answers = [""] * len(questions)

    idx = st.session_state.get("current_question", 0)
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if idx > 0:
            if st.button("Previous"):
                st.session_state.current_question = idx - 1
                st.rerun()
    with col3:
        if idx < len(questions) - 1:
            if st.button("Next"):
                st.session_state.current_question = idx + 1
                st.rerun()

    # Question Display
    st.subheader(f"Question {idx + 1}/{len(questions)}")
    current_q = questions[idx]
    st.markdown(f"**{current_q['question']}**")

    # Audio of current question from user-specific audio directory
    audio_dir = os.path.join(USER_DATA_DIR, username)
    audio_path = os.path.join(audio_dir, f"audio/q{idx+1}.wav")
    if os.path.exists(audio_path):
        st.audio(audio_path, format='audio/wav', autoplay=True)
    else:
        st.warning("Audio missing for this question")
    
    # For coding questions, show input/output examples and code editor
    if current_q.get('type') == 'coding':
        if 'input_example' in current_q:
            st.markdown(f"*Input Example:* `{current_q['input_example']}`")
        if 'output_example' in current_q:
            st.markdown(f"*Expected Output:* `{current_q['output_example']}`")
        
        st.title("Code Editor")
        language = st.selectbox("Select Language", ["python", "c", "java"])
        theme = st.selectbox("Select Theme", ["chrome", "twilight", "github"])
        font_size = st.slider("Font Size", 10, 24, 18)
        code = st_ace(
            placeholder="Write your code here...",
            language=language,
            theme=theme,
            key=f"editor_{language}",
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
            st.code(code, language=language)
            st.session_state.answers[idx] = code if code.strip() != "" else "NA"
    
    # For conceptual questions: recording and/or text answer
    if current_q.get('type') != 'coding':
        st.markdown("### Your Answer")
        wave_byets = st_audiorec()
        
        if wave_byets is not None:
            file_path, transcription = generate_text_from_audio(wave_byets)
            if transcription:
                st.session_state.answers[idx] = transcription
                st.success("Transcription received!")
        else:
            transcription = st.session_state.answers[idx]
        answer_text = st.text_area("Answer will appear here", transcription if transcription else "")
        if answer_text:
            st.session_state.answers[idx] = answer_text 
        else:
            st.session_state.answers[idx] = "NA"

    if st.button("Submit All Answers"):
        for i in range(len(questions)):
            questions[i]['user_answer'] = st.session_state.answers[i]
        
        filename  = os.path.join(USER_DATA_DIR, f"{username}/questions.json")
        with open(filename, 'w') as f:
            json.dump({"questions": questions}, f)
        
        st.success("Answers submitted successfully!")

# Page 3: Display Results and Delete Audio Files for this user
def page_view_results():
    st.header("Interview Results")
    username = st.text_input("Enter your username:")
    evaluate= st.button("Evaluate Answers")
    if username and evaluate:
            result = evaluate_answers(username)
            
            if result:
                st.subheader("Final Result")
                st.markdown(f"""
                **Total Points:** {result['total_points']}  
                **Percentage:** {result['percentage']:.2f}%  
                **Status:** {result['status']}
                """)
                
                st.subheader("Detailed Evaluation")
                questions = load_questions(username)
                for i, q in enumerate(questions, 1):
                    st.markdown(f"**Q{i}:** {q['question']}")
                    if q['type'] == "coding":
                        st.markdown("**Your Answer:**")
                        st.code(f"{q.get('user_answer', 'No answer')}")
                        st.markdown(f"**passed_test-cases:** {q.get('passed_cases', 'No feedback available')}")
                        st.markdown(f"**total_test-cases:** {q.get('total_cases', 'No feedback available')}")
                    else:
                        st.markdown(f"**Your Answer:** {q.get('user_answer', 'No answer')}")
                    st.markdown(f"**Score:** {q.get('points', 0)}/10")
                    st.markdown(f"**Feedback:** {q.get('feedback', 'No feedback available')}")
                    st.divider()
            
            # Delete the user's audio directory when the result page is viewed.
            audio_dir = os.path.join(USER_DATA_DIR, f"{username}/audio")
            if os.path.exists(audio_dir):
                # Remove all files inside and then the directory
                for filename in os.listdir(audio_dir):
                    file_path = os.path.join(audio_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(audio_dir)
                st.success("User-specific audio files have been deleted.")

# Main App Navigation
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
