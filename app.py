import spacy
import gradio as gr
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define skill keywords (expand as needed)
keywords = ["java", "sql", "spring boot", "python", "pandas", "qwen", "mistral", 
            "llm", "artificial intelligence", "machine learning"]

# Build PhraseMatcher for multi-word skills
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in keywords]
matcher.add("SKILLS", patterns)

def resume_analyzer(resume_text):
    doc = nlp(resume_text)
    matches = matcher(doc)
    skills = list(set([doc[start:end].text.title() for match_id, start, end in matches]))
    return skills

def interview_system(resume, *answers):
    skills = resume_analyzer(resume)

    technical_questions = [f"Explain {skill} in detail." for skill in skills]
    behavioral_questions = [
        "Tell me about a time you solved a difficult problem.",
        "Describe a situation where you worked in a team."
    ]

    # Collect answers dynamically
    answers_dict = {skill: ans for skill, ans in zip(skills, answers[:len(skills)])}
    answers_dict["Behavior1"] = answers[-2] if len(answers) >= 2 else ""
    answers_dict["Behavior2"] = answers[-1] if len(answers) >= 1 else ""

    # Simple scoring: length of answers
    score = sum(len(ans.strip()) for ans in answers_dict.values()) // 10
    feedback = "Good job! Work on refining behavioral answers." if score > 5 else "Focus on expanding technical answers with examples."

    return skills, technical_questions, behavioral_questions, score, feedback, "Would you like to retry your weakest answer?"

# Example skills for UI preview (dynamic rebuild would be next step)
skills_example = ["1", "2"]  # Replace with resume_analyzer output if you want auto-rebuild

# Build dynamic inputs
inputs = [gr.Textbox(label=f"Answer: {skill}") for skill in skills_example]
inputs += [
    gr.Textbox(label="Answer: Behavioral Q1"),
    gr.Textbox(label="Answer: Behavioral Q2")
]

demo = gr.Interface(
    fn=interview_system,
    inputs=[gr.Textbox(label="Paste Resume")] + inputs,
    outputs=[
        gr.Textbox(label="Extracted Skills"),
        gr.Textbox(label="Technical Questions"),
        gr.Textbox(label="Behavioral Questions"),
        gr.Number(label="Score"),
        gr.Textbox(label="Feedback"),
        gr.Textbox(label="Reflection")
    ],
    title="Multi-Agent Interview Preparation System"
)

demo.launch(share=True, debug=True)
