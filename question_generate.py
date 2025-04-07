import os
import json
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

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

def evaluate_answers():
    try:
        with open('interview_data.json', 'r') as f:
            data = json.load(f)
            questions = data.get("questions", [])
            
            total_points = 0
            max_points = len(questions) * 10  # Assuming 10 points per question
            
            for q in questions:
                # Simple evaluation logic - compare with sample answer
                # You can implement more sophisticated evaluation
                if q['user_answer'].lower() in q['answer'].lower():
                    q['points'] = 10
                    total_points += 10
                else:
                    q['points'] = 5  # Partial credit
                    total_points += 5

            data['result'] = {
                'total_points': total_points,
                'percentage': (total_points / max_points) * 100,
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
    
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button("Generate Questions")
    
    # Only show regenerate button if questions have been generated before
    regenerate_btn = False
    if st.session_state.generated:
        with col2:
            regenerate_btn = st.button("Regenerate Questions")
    
    if generate_btn or regenerate_btn:
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
                st.success("Questions generated successfully!" if generate_btn 
                         else "Questions regenerated successfully!")
                st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
            else:
                st.error("Failed to generate questions")
        else:
            st.warning("Please enter a job description")

    if st.session_state.generated:
        st.subheader("Generated Questions")
        questions = st.session_state.get("questions", load_questions())
        
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}:** {q['question']}")
            st.markdown(f"**Sample Answer:** {q['answer']}")
            st.divider()
# def page_generate_questions():
    # st.header("Generate Interview Questions")
    # job_desc = st.text_area("Paste Job Description Here", height=200)
    
    # if st.button("Generate Questions"):
    #     if job_desc.strip():
    #         questions = generate_interview_questions(job_desc)
    #         if questions:
    #             save_questions(questions)
    #             st.session_state.generated = True
    #             st.success("Questions generated successfully!")
                
    #             st.subheader("Generated Questions")
    #             for i, q in enumerate(questions, 1):
    #                 st.markdown(f"**Q{i}:** {q['question']}")
    #                 st.markdown(f"**Sample Answer:** {q['answer']}")
    #         else:
    #             st.error("Failed to generate questions")
    #     else:
    #         st.warning("Please enter a job description")


# Page 2: Answer Questions
def page_answer_questions():
    st.header("Answer Interview Questions")
    questions = load_questions()
    
    if not questions:
        st.warning("No questions found. Generate questions first!")
        return
    
    total_questions = len(questions)
    
    # Initialize answers if not exists
    if len(st.session_state.answers) != total_questions:
        st.session_state.answers = [""] * total_questions
    
    # Navigation controls
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.session_state.current_question > 0:
            if st.button("Previous"):
                st.session_state.current_question -= 1
    with col3:
        if st.session_state.current_question < total_questions - 1:
            if st.button("Next"):
                st.session_state.current_question += 1
    
    # Display current question
    idx = st.session_state.current_question
    st.subheader(f"Question {idx + 1}/{total_questions}")
    st.markdown(f"**{questions[idx]['question']}**")
    
    # Answer input
    st.session_state.answers[idx] = st.text_area(
        "Your Answer",
        value=st.session_state.answers[idx],
        key=f"answer_{idx}"
    )
    
    # Submit all answers
    if all(st.session_state.answers):
        if st.button("Submit All Answers"):
            # Update questions with user answers
            for i in range(total_questions):
                questions[i]['user_answer'] = st.session_state.answers[i]
            
            with open('interview_data.json', 'w') as f:
                json.dump({"questions": questions}, f)
            
            st.success("Answers submitted successfully!")

# Page 3: View Results
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
                st.markdown(f"**Points:** {q.get('points', 0)}/10")
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