#!/usr/bin/env python
# coding: utf-8

# In[83]:


import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
from spacy.matcher import Matcher
import re
import io
import base64
import time
import os
from datetime import datetime
from PIL import Image
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[84]:


# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# In[85]:


# Initialize spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Define skill databases for different job roles
data_science_skills = [
    'python', 'r', 'sql', 'machine learning', 'deep learning', 'tensorflow',
    'keras', 'pytorch', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib',
    'seaborn', 'data visualization', 'statistics', 'data mining', 'big data',
    'hadoop', 'spark', 'nlp', 'natural language processing', 'computer vision',
    'feature engineering', 'dimensionality reduction', 'clustering', 'classification',
    'regression', 'flask', 'streamlit', 'tableau', 'power bi', 'excel', 'data analysis'
]

web_dev_skills = [
    'html', 'css', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node.js',
    'express', 'django', 'flask', 'php', 'laravel', 'ruby', 'rails', 'asp.net',
    'bootstrap', 'jquery', 'rest api', 'graphql', 'json', 'xml', 'webpack', 'git',
    'responsive design', 'web security', 'sql', 'nosql', 'mongodb', 'mysql',
    'postgresql', 'aws', 'azure', 'docker', 'kubernetes', 'ci/cd'
]

android_dev_skills = [
    'java', 'kotlin', 'android studio', 'xml', 'json', 'material design', 'android sdk',
    'rest api', 'sqlite', 'room', 'jetpack', 'mvvm', 'mvp', 'firebase', 'google maps api',
    'location services', 'notifications', 'fragments', 'activities', 'intents', 'gradle',
    'git', 'junit', 'espresso', 'rx java', 'dagger', 'retrofit', 'glide', 'picasso'
]

ios_dev_skills = [
    'swift', 'objective-c', 'xcode', 'uikit', 'core data', 'core animation',
    'auto layout', 'cocoa touch', 'cocoapods', 'grand central dispatch', 'rest api',
    'json', 'alamofire', 'push notifications', 'core location', 'mapkit', 'arkit',
    'healthkit', 'storyboard', 'swiftui', 'combine', 'unit testing', 'ui testing'
]

uiux_skills = [
    'figma', 'sketch', 'adobe xd', 'illustrator', 'photoshop', 'prototyping',
    'wireframing', 'user research', 'usability testing', 'information architecture',
    'interaction design', 'visual design', 'typography', 'color theory', 'accessibility',
    'responsive design', 'design systems', 'user flows', 'personas', 'journey mapping',
    'a/b testing', 'material design', 'ios design guidelines'
]

# Courses recommendations
ds_courses = [
    {'name': 'Machine Learning by Andrew NG', 'url': 'https://www.coursera.org/learn/machine-learning'},
    {'name': 'Python for Data Science and Machine Learning Bootcamp', 'url': 'https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/'},
    {'name': 'Deep Learning Specialization', 'url': 'https://www.coursera.org/specializations/deep-learning'},
    {'name': 'Applied Data Science with Python', 'url': 'https://www.coursera.org/specializations/data-science-python'},
    {'name': 'TensorFlow Developer Certificate', 'url': 'https://www.tensorflow.org/certificate'},
    {'name': 'Data Science: Foundations using R', 'url': 'https://www.coursera.org/specializations/data-science-foundations-r'}
]

web_courses = [
    {'name': 'The Web Developer Bootcamp', 'url': 'https://www.udemy.com/course/the-web-developer-bootcamp/'},
    {'name': 'JavaScript Algorithms and Data Structures', 'url': 'https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/'},
    {'name': 'React - The Complete Guide', 'url': 'https://www.udemy.com/course/react-the-complete-guide-incl-redux/'},
    {'name': 'The Complete Node.js Developer Course', 'url': 'https://www.udemy.com/course/the-complete-nodejs-developer-course-2/'},
    {'name': 'Full Stack Web Development with Django', 'url': 'https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp/'}
]

android_courses = [
    {'name': 'Android Basics in Kotlin', 'url': 'https://developer.android.com/courses/android-basics-kotlin/course'},
    {'name': 'Advanced Android in Kotlin', 'url': 'https://developer.android.com/courses/kotlin-android-advanced/overview'},
    {'name': 'The Complete Android 12 & Kotlin Development Masterclass', 'url': 'https://www.udemy.com/course/android-kotlin-developer/'},
    {'name': 'Android Architecture Components', 'url': 'https://developer.android.com/codelabs/android-room-with-a-view'},
    {'name': 'Android App Development Specialization', 'url': 'https://www.coursera.org/specializations/android-app-development'}
]

ios_courses = [
    {'name': 'iOS & Swift - The Complete iOS App Development Bootcamp', 'url': 'https://www.udemy.com/course/ios-13-app-development-bootcamp/'},
    {'name': 'SwiftUI Masterclass', 'url': 'https://www.udemy.com/course/swiftui-masterclass-course-ios-development-with-swift/'},
    {'name': 'iOS App Development with Swift Specialization', 'url': 'https://www.coursera.org/specializations/app-development'},
    {'name': 'Develop in Swift', 'url': 'https://www.apple.com/education/k12/teaching-code/#develop-in-swift'},
    {'name': 'Learn Swift by Apple', 'url': 'https://swift.org/documentation/#the-swift-programming-language'}
]

uiux_courses = [
    {'name': 'Google UX Design Professional Certificate', 'url': 'https://www.coursera.org/professional-certificates/google-ux-design'},
    {'name': 'UI / UX Design Specialization', 'url': 'https://www.coursera.org/specializations/ui-ux-design'},
    {'name': 'User Experience Research and Design', 'url': 'https://www.coursera.org/specializations/michiganux'},
    {'name': 'Interaction Design Specialization', 'url': 'https://www.coursera.org/specializations/interaction-design'},
    {'name': 'The Complete App Design Course', 'url': 'https://www.udemy.com/course/the-complete-app-design-course-ux-and-ui-design/'}
]


# In[86]:


# Resume parsing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# In[87]:


def extract_name(resume_text):
    """Extract name from resume text using improved methods"""
    lines = resume_text.split('\n')
    
    # First strategy: Look for standalone name in all caps (most common for personal name)
    for line in lines:
        line = line.strip()
        # Names are typically shorter lines, all caps, without common title prefixes
        if line.isupper() and 5 <= len(line) <= 30 and ' ' in line:
            words = line.split()
            # Filter out lines that are section headers
            if not any(header in line.upper() for header in ['PERSONAL DETAILS', 'EDUCATION', 'SUMMARY', 'SKILLS']):
                # Filter out lines containing "Mrs." or "Mr."
                if not any(title in line for title in ["MRS", "MR", "MISS", "MS"]):
                    return line
    
    # Second strategy: Try spaCy NER, but filter out names with titles
    doc = nlp(resume_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_name = ent.text
            if not any(title in person_name for title in ["Mrs.", "Mr.", "Miss", "Ms."]):
                return person_name
    
    # Third strategy: Check first few lines for potential names
    for line in lines[:15]:  # Check more lines
        if line.strip() and len(line.strip().split()) <= 3:  # Name likely has few words
            # Skip lines with common headers or titles
            if any(header in line.lower() for header in ['resume', 'cv', 'personal details', 'profile', 'mrs', 'mr.']):
                continue
            return line.strip()
    
    return "Name not found"


# In[88]:


def extract_email(resume_text):
    """Extract email from resume text using regex"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, resume_text)
    return emails[0] if emails else "Email not found"

def extract_phone(resume_text):
    """Extract phone number from resume text using regex with improved pattern"""
    # More flexible pattern for international and local formats
    phone_patterns = [
        r'(\+\d{1,3}[-.\\s]?)?(\d{3})[-.\\s]?(\d{3})[-.\\s]?(\d{4})',  # +91 999 999 9999 format
        r'(\d{3})[-.\\s]?(\d{3})[-.\\s]?(\d{4})',  # 999 999 9999 format
        r'(\d{5})[-.\\s]?(\d{5})',  # 99999 99999 format (common in some countries)
        r'Phone:?\s*(.+?)(?=\n|$)'  # Look for "Phone: " followed by number
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, resume_text)
        if matches:
            # For the "Phone:" pattern, return the captured group
            if pattern.startswith('Phone'):
                return matches[0].strip()
            # For other patterns, reconstruct the phone number
            match = matches[0]
            if isinstance(match, tuple):
                # Join non-empty parts of the match
                return ''.join([part for part in match if part])
            return match
    
    return "Phone not found"

def extract_skills(resume_text):
    """Extract skills from resume text"""
    # Convert to lowercase and tokenize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Use simple split instead of word_tokenize to avoid punkt_tab dependency
    words = resume_text.lower().split()
    filtered_tokens = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]
    
    # Combine all skill categories for matching
    all_skills = data_science_skills + web_dev_skills + android_dev_skills + ios_dev_skills + uiux_skills
    all_skills = list(set(all_skills))  # Remove duplicates
    
    # Match skills in resume
    matched_skills = []
    for token in filtered_tokens:
        if token in all_skills:
            matched_skills.append(token)
    
    # Also check for bigrams and trigrams for multi-word skills
    text = ' '.join(filtered_tokens)
    for skill in all_skills:
        if ' ' in skill and skill in text:
            matched_skills.append(skill)
    
    return list(set(matched_skills))  # Remove duplicates


# In[89]:


def predict_job_field(skills):
    """Predict job field based on matched skills"""
    field_scores = {
        'Data Science': 0,
        'Web Development': 0,
        'Android Development': 0,
        'iOS Development': 0,
        'UI/UX Design': 0
    }
    
    for skill in skills:
        if skill in data_science_skills:
            field_scores['Data Science'] += 1
        if skill in web_dev_skills:
            field_scores['Web Development'] += 1
        if skill in android_dev_skills:
            field_scores['Android Development'] += 1
        if skill in ios_dev_skills:
            field_scores['iOS Development'] += 1
        if skill in uiux_skills:
            field_scores['UI/UX Design'] += 1
    
    # Return field with highest score
    if not field_scores or max(field_scores.values()) == 0:
        return "Unknown"
    
    return max(field_scores, key=field_scores.get)


# In[90]:


def get_recommended_skills(job_field, current_skills):
    """Get recommended skills based on job field and current skills"""
    if job_field == 'Data Science':
        all_relevant_skills = data_science_skills
    elif job_field == 'Web Development':
        all_relevant_skills = web_dev_skills
    elif job_field == 'Android Development':
        all_relevant_skills = android_dev_skills
    elif job_field == 'iOS Development':
        all_relevant_skills = ios_dev_skills
    elif job_field == 'UI/UX Design':
        all_relevant_skills = uiux_skills
    else:
        return []
    
    # Return skills not already in current_skills
    return [skill for skill in all_relevant_skills if skill not in current_skills][:10]


# In[91]:


def get_recommended_courses(job_field):
    """Get recommended courses based on job field"""
    if job_field == 'Data Science':
        return ds_courses
    elif job_field == 'Web Development':
        return web_courses
    elif job_field == 'Android Development':
        return android_courses
    elif job_field == 'iOS Development':
        return ios_courses
    elif job_field == 'UI/UX Design':
        return uiux_courses
    else:
        return []


# In[92]:


def calculate_resume_score(resume_text, skills):
    """Calculate resume score based on various factors"""
    score = 0
    
    # More skills = higher score (max 30 points)
    skill_count = len(skills)
    score += min(skill_count * 3, 30)
    
    # Check for important resume sections (max 40 points)
    sections = {
        'education': ['education', 'academic', 'degree', 'university', 'college', 'school'],
        'experience': ['experience', 'work', 'job', 'employment', 'career'],
        'projects': ['project', 'portfolio', 'github'],
        'skills': ['skill', 'proficiency', 'competency'],
        'achievements': ['achievement', 'award', 'honor', 'scholarship'],
        'objective': ['objective', 'summary', 'profile', 'about']
    }
    
    for section, keywords in sections.items():
        for keyword in keywords:
            if keyword in resume_text.lower():
                score += (40 / len(sections))
                break
    
    # Check resume length (10 points)
    word_count = len(resume_text.split())
    if 300 <= word_count <= 700:
        score += 10
    elif word_count > 700:
        score += 5
    else:
        score += word_count / 60  # Partial credit for shorter resumes
    
    # Check for contact info (10 points)
    if extract_email(resume_text) != "Email not found":
        score += 5
    if extract_phone(resume_text) != "Phone not found":
        score += 5
    
    # Check for action verbs (10 points)
    action_verbs = ['achieved', 'improved', 'developed', 'created', 'implemented', 'managed',
                    'designed', 'analyzed', 'built', 'led', 'increased', 'decreased', 'solved',
                    'delivered', 'conducted', 'organized', 'coordinated', 'generated', 'produced']
    
    action_verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
    score += min(action_verb_count, 10)
    
    return min(round(score), 100)  # Cap at 100


# In[93]:


def get_pdf_download_link(text, filename):
    """Generate a link to download the text as a PDF file"""
    html = f"""
    <a href="data:application/octet-stream;base64,{base64.b64encode(text.encode()).decode()}" 
       download="{filename}">Download Resume Analysis as PDF</a>
    """
    return html


# In[94]:


def main():
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("AI Resume Analyzer")
    st.markdown("Upload your resume and get AI-powered analysis and recommendations")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Resume Analysis", "About"])
    
    if app_mode == "About":
        st.markdown("# About AI Resume Analyzer")
        st.markdown("""
        This application helps you analyze your resume, extract important information, 
        and provides personalized recommendations to improve your chances of getting hired.
        
        ### Features:
        - Extract key information from your resume
        - Identify your skills and predict suitable job roles
        - Calculate your resume score
        - Recommend skills to add to your resume
        - Suggest relevant courses for skill development
        
        ### How to use:
        1. Upload your resume in PDF format
        2. Get instant analysis and recommendations
        3. Use the feedback to improve your resume
        """)
        
    else:
        # Main page
        uploaded_file = st.file_uploader("Choose your Resume (PDF format)", type=["pdf"])
        
        if uploaded_file is not None:
            with st.spinner("Analyzing your resume..."):
                # Save the file temporarily
                with open("temp_resume.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # Display the PDF
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### Your Resume")
                    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Resume Analysis")
                    
                    # Extract information
                    name = extract_name(resume_text)
                    email = extract_email(resume_text)
                    phone = extract_phone(resume_text)
                    skills = extract_skills(resume_text)
                    job_field = predict_job_field(skills)
                    
                    # Basic info section
                    st.subheader("Basic Information")
                    st.write(f"**Name:** {name}")
                    st.write(f"**Email:** {email}")
                    st.write(f"**Phone:** {phone}")
                    
                    # Calculate score
                    resume_score = calculate_resume_score(resume_text, skills)
                    
                    # Skills section
                    st.subheader("Skills Analysis")
                    st.write(f"**Detected Job Field:** {job_field}")
                    
                    if skills:
                        st.write("**Skills found in your resume:**")
                        skill_columns = st.columns(3)
                        for i, skill in enumerate(skills):
                            skill_columns[i % 3].markdown(f"- {skill.title()}")
                    else:
                        st.warning("No skills were detected. Make sure your skills are clearly listed in your resume.")
                    
                    # Resume Score
                    st.subheader("Resume Score")
                    st.write(f"Your resume scores **{resume_score}%** based on our analysis.")
                    st.progress(resume_score/100)
                    
                    score_interpretation = ""
                    if resume_score < 50:
                        score_interpretation = "Your resume needs significant improvement."
                    elif resume_score < 70:
                        score_interpretation = "Your resume is average and could use some enhancements."
                    else:
                        score_interpretation = "Your resume is strong, with just a few suggestions for improvement."
                    
                    st.write(score_interpretation)
                
                # Recommendations section
                st.markdown("### Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Skills to Add")
                    recommended_skills = get_recommended_skills(job_field, skills)
                    
                    if recommended_skills:
                        st.write("Adding these skills could improve your resume for this job field:")
                        for skill in recommended_skills:
                            st.markdown(f"- {skill.title()}")
                    else:
                        st.write("You already have a comprehensive set of skills for this field!")
                
                with col2:
                    st.subheader("Resume Improvement Tips")
                    
                    # Resume tips based on score
                    improvement_areas = []
                    
                    # Check for common sections
                    if 'objective' not in resume_text.lower() and 'summary' not in resume_text.lower():
                        improvement_areas.append("Add a clear objective or professional summary")
                    
                    if 'experience' not in resume_text.lower() and 'work' not in resume_text.lower():
                        improvement_areas.append("Include your work experience with measurable achievements")
                    
                    if 'education' not in resume_text.lower():
                        improvement_areas.append("Add your educational background")
                    
                    if 'project' not in resume_text.lower():
                        improvement_areas.append("Highlight relevant projects you've worked on")
                    
                    # Check for action verbs
                    action_verbs = ['achieved', 'improved', 'developed', 'created', 'implemented']
                    if not any(verb in resume_text.lower() for verb in action_verbs):
                        improvement_areas.append("Use more action verbs to describe your achievements")
                    
                    # Check for skills section
                    if 'skill' not in resume_text.lower():
                        improvement_areas.append("Create a dedicated skills section")
                    
                    if improvement_areas:
                        for area in improvement_areas:
                            st.markdown(f"- {area}")
                    else:
                        st.write("Your resume is well-structured!")
                
                # Course recommendations
                st.subheader("Recommended Courses")
                recommended_courses = get_recommended_courses(job_field)
                
                if recommended_courses:
                    course_cols = st.columns(3)
                    for i, course in enumerate(recommended_courses[:6]):  # Show up to 6 courses
                        with course_cols[i % 3]:
                            st.markdown(f"**{course['name']}**")
                            st.markdown(f"[Go to Course]({course['url']})")
                else:
                    st.write("No specific courses to recommend for this field.")
                
                # Generate a downloadable report
                st.subheader("Download Analysis")
                
                report = f"""
                AI RESUME ANALYZER REPORT
                -------------------------
                
                BASIC INFORMATION:
                Name: {name}
                Email: {email}
                Phone: {phone}
                
                SKILLS ANALYSIS:
                Detected Job Field: {job_field}
                Skills Found: {', '.join(skills)}
                
                RESUME SCORE: {resume_score}%
                {score_interpretation}
                
                RECOMMENDATIONS:
                Skills to Add: {', '.join(recommended_skills)}
                
                Improvement Areas:
                {chr(10).join('- ' + area for area in improvement_areas)}
                
                RECOMMENDED COURSES:
                {chr(10).join('- ' + course['name'] + ': ' + course['url'] for course in recommended_courses[:5])}
                
                Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="Download Analysis Report",
                    data=report,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain"
                )
                
                # Clean up
                try:
                    os.remove("temp_resume.pdf")
                except:
                    pass

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




