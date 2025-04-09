import os
import json
from groq import Groq
from dotenv import load_dotenv
import streamlit as st
import tempfile
import soundfile as sf
from kokoro import KPipeline
import pyaudio
import wave
import whisper
import numpy as np
import time
import threading


# Audio Configuration
FORMAT = pyaudio.paInt16
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

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

def transcribe_audio(file_path):
    """Convert speech to text using Whisper"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        model = load_whisper_model()
        result = model.transcribe(file_path, fp16=False, language="en")
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

# Updated Recording Function with Threading
# Updated audio recording functions
def record_user_response():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("streamstream", stream)

    frames = []
    start_time = time.time()
    last_voice_time = start_time

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_chunk).mean()
        current_time = time.time()

        print("audio_chunkaudio_chunk", audio_chunk)
        if volume > SILENCE_THRESHOLD:
            last_voice_time = current_time
            frames.append(data)
        else:
            frames.append(data)

        if (current_time - last_voice_time) > SILENCE_TIMEOUT:
            break
        elif (current_time - start_time) > MAX_RECORD_TIME:
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(TEMP_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return TEMP_FILENAME

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
    system_prompt = """You are a hiring manager expert. Generate 6 technical interview questions 
    based on the job description. For each question, provide a sample answer. 
    Return format: JSON array with {question, answer} objects."""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": job_description}
            ],
            model=model,
            response_format={"type": "json_object"}
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
    system_prompt = """You are an expert technical interviewer. Analyze how well the user's answer matches the 
    sample answer in terms of:
    1. Technical accuracy
    2. Key concepts covered
    3. Depth of understanding
    4. Relevance to the question

    ** Focus on semantic similarity rather than exact wording. **
    Return JSON format: {
        "score": 0-10 (10=excellent match),
        "feedback": "brief constructive feedback"
    }
    feedback answer **Never** contain lines like "response perfectly matches the sample answer","almost identical to the sample answer" 
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
                user_answer = q.get('user_answer', '').strip()
                sample_answer = q.get('answer', '').strip()
                
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


# Modified Answer Page
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
    st.markdown(f"**{questions[idx]['question']}**")
    
    # Audio playback
    audio_path = os.path.join("question_audios", f"q{idx+1}.wav")
    if os.path.exists(audio_path):
        st.audio(audio_path, format='audio/wav')
    else:
        st.warning("Audio missing for this question")

    # Voice Answer Section
    st.markdown("### Your Answer")
    
    # Recording controls
    col1, col2 = st.columns([4, 1])
    
    with col1:
        answer_text = st.text_area(
            "Answer will appear here",
            value=st.session_state.answers[idx],
            key=f"answer_{idx}",
            height=150
        )
    
    # Updated answer section in page_answer_questions()
    with col2:
        if not st.session_state.recording:
            if st.button("🎤 Start Recording", key=f"start_{idx}"):
                st.session_state.recording = True
                st.session_state.stop_event.clear()
                st.session_state.recording_thread = threading.Thread(target=record_user_response)
                st.session_state.recording_thread.start()
        else:
            if st.button("⏹️ Stop Recording", key=f"stop_{idx}"):
                st.session_state.recording = False
                st.session_state.stop_event.set()
                st.session_state.recording_thread.join()
                
                # Get the recorded file path
                file_path = os.path.abspath(TEMP_FILENAME)
                
                if os.path.exists(file_path):
                    # Add loading spinner
                    with st.spinner("Transcribing audio..."):
                        transcribed_text = transcribe_audio(file_path)
                        if transcribed_text:
                            st.session_state.answers[idx] = transcribed_text
                            st.rerun()
                        else:
                            st.error("Failed to transcribe audio")
                    
                    # Clean up the audio file
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Error cleaning up audio file: {str(e)}")
                else:
                    st.error("Recording file not found")

    # Store answer in session state
    st.session_state.answers[idx] = answer_text

    # Submission Section
    all_answered = all(st.session_state.answers)
    
    if st.button("Submit All Answers", disabled=not all_answered):
        for i in range(len(questions)):
            questions[i]['user_answer'] = st.session_state.answers[i]
        
        with open('interview_data.json', 'w') as f:
            json.dump({"questions": questions}, f)
        
        st.success("Answers submitted successfully!")

    # Progress indicator
    answered_count = sum(1 for ans in st.session_state.answers if ans.strip())
    st.progress(answered_count / len(questions))
    st.caption(f"Answered {answered_count}/{len(questions)} questions")


# Update the results display to show feedback
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