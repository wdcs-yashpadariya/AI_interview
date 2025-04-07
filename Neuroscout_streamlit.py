import os
import json
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize session state variables
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

def generate_interview_questions(job_description, model="llama3-70b-8192"):
    system_prompt = """You are a hiring manager expert. Generate 5 technical interview questions 
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

def evaluate_answers(questions, user_answers):
    evaluation = []
    for i, (question, user_answer) in enumerate(zip(questions, user_answers)):
        prompt = f"""
        Question: {question['question']}
        Model Answer: {question['answer']}
        Candidate Answer: {user_answer}
        
        Evaluate the candidate's answer on:
        1. Technical accuracy (0-10)
        2. Completeness (0-10)
        3. Clarity (0-10)
        Return as JSON with scores and brief feedback.
        """
        
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                response_format={"type": "json_object"}
            )
            evaluation.append(json.loads(response.choices[0].message.content))
        except Exception as e:
            st.error(f"Error evaluating answer {i+1}: {str(e)}")
            evaluation.append({"error": str(e)})
    
    return evaluation

def main():
    st.title("Interactive Interview System")
    
    if not st.session_state.questions:
        # First step: Get job description and generate questions
        st.header("Step 1: Enter Job Description")
        job_description = st.text_area("Paste the job description here:", height=150)
        
        if st.button("Generate Questions"):
            if job_description.strip():
                with st.spinner("Generating interview questions..."):
                    st.session_state.questions = generate_interview_questions(job_description)
                    st.session_state.user_answers = [""] * len(st.session_state.questions)
                    st.rerun()
            else:
                st.warning("Please enter a job description")
    else:
        if not st.session_state.show_results:
            # Second step: Conduct the interview
            st.header(f"Question {st.session_state.current_question + 1}/{len(st.session_state.questions)}")
            
            current_q = st.session_state.questions[st.session_state.current_question]
            st.markdown(f"**{current_q['question']}**")
            
            # Text area for user's answer
            st.session_state.user_answers[st.session_state.current_question] = st.text_area(
                "Your answer:", 
                value=st.session_state.user_answers[st.session_state.current_question],
                height=200,
                key=f"answer_{st.session_state.current_question}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.current_question > 0:
                    if st.button("Previous Question"):
                        st.session_state.current_question -= 1
                        st.rerun()
            
            with col2:
                if st.session_state.current_question < len(st.session_state.questions) - 1:
                    if st.button("Next Question"):
                        st.session_state.current_question += 1
                        st.rerun()
                else:
                    if st.button("Submit All Answers"):
                        st.session_state.show_results = True
                        st.rerun()
        else:
            # Third step: Show evaluation results
            st.header("Interview Results")
            
            with st.spinner("Evaluating your answers..."):
                evaluation = evaluate_answers(st.session_state.questions, st.session_state.user_answers)
                
                total_score = 0
                max_score = len(st.session_state.questions) * 30  # 10 points per category x 3 categories
                
                for i, (q, eval_result) in enumerate(zip(st.session_state.questions, evaluation)):
                    st.subheader(f"Question {i+1}: {q['question']}")
                    st.markdown(f"**Your Answer:** {st.session_state.user_answers[i]}")
                    
                    if 'error' in eval_result:
                        st.error(f"Evaluation error: {eval_result['error']}")
                    else:
                        st.markdown(f"""
                        - Technical Accuracy: {eval_result.get('technical_accuracy', 'N/A')}/10
                        - Completeness: {eval_result.get('completeness', 'N/A')}/10
                        - Clarity: {eval_result.get('clarity', 'N/A')}/10
                        """)
                        st.markdown(f"**Feedback:** {eval_result.get('feedback', 'No feedback available')}")
                        
                        # Calculate total score
                        total_score += sum([
                            eval_result.get('technical_accuracy', 0),
                            eval_result.get('completeness', 0),
                            eval_result.get('clarity', 0)
                        ])
                
                # Determine pass/fail (adjust threshold as needed)
                pass_threshold = max_score * 0.7  # 70% score to pass
                st.divider()
                st.subheader("Final Result")
                st.markdown(f"**Total Score:** {total_score}/{max_score}")
                
                if total_score >= pass_threshold:
                    st.success("✅ Congratulations! You passed the interview!")
                else:
                    st.error("❌ Unfortunately, you didn't meet the passing criteria")
            
            if st.button("Start New Interview"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()