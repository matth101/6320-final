import PyPDF2
import docx
import re
import spacy
from nltk.corpus import stopwords
from collections import Counter

import nltk
nltk.data.path.append('C:/Users/MrRic/AppData/Roaming/nltk_data')
nltk.download('punkt_tab', quiet=True)

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

try:
    nlp = spacy.load('en_core_web_sm')
except:
    import sys
    print("You need to download the spaCy model. Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text(file_path):
    """Extract text from PDF or DOCX file"""
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide PDF or DOCX.")

def extract_skills(text):
    """Extract skills from text using NLP techniques"""
    # Common skill keywords
    technical_skills = [
        "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", "kotlin", "golang",
        "html", "css", "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "firebase",
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "bitbucket",
        "react", "angular", "vue", "node.js", "django", "flask", "spring", ".net", "express",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "tableau", 
        "power bi", "excel", "spss", "r", "machine learning", "deep learning", "nlp", "ai",
        "data analysis", "data science", "data visualization", "data mining", "data modeling",
        "data engineering", "etl", "data warehousing", "big data", "hadoop", "spark", "kafka",
        "agile", "scrum", "kanban", "jira", "confluence", "trello", "asana", "slack",
        "rest api", "graphql", "json", "xml", "soap", "microservices", "devops", "ci/cd",
        "test automation", "selenium", "cypress", "jest", "mocha", "junit", "pytest",
        "tdd", "bdd", "uml", "erd", "wireframing", "figma", "sketch", "adobe xd", "photoshop",
        "illustrator", "indesign", "premiere pro", "after effects", "final cut pro",
        "project management", "product management", "program management", "leadership", "team lead"
    ]
    
    soft_skills = [
        "communication", "teamwork", "problem solving", "critical thinking", "creativity",
        "time management", "organization", "adaptability", "flexibility", "leadership",
        "interpersonal", "presentation", "negotiation", "conflict resolution", "decision making",
        "emotional intelligence", "stress management", "work ethic", "attention to detail",
        "customer service", "analytical thinking", "strategic planning", "innovation", "mentoring"
    ]
    
    all_skills = technical_skills + soft_skills
    
    text_lower = text.lower()
    
    found_skills = []
    for skill in all_skills:
        # Match whole words only using regex
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"] and ent.text.lower() not in found_skills:
            # Check if it might be a technical product/tool/framework
            if any(tech in ent.text.lower() for tech in ["framework", "language", "tool", "platform", "software", "system"]):
                found_skills.append(ent.text.lower())
    
    return found_skills

def extract_education(text):
    education_keywords = [
        "bachelor", "master", "phd", "doctorate", "mba", "bs", "ms", "ba", "ma", "b.s.", "m.s.", 
        "b.a.", "m.a.", "ph.d.", "degree", "university", "college", "institute", "school"
    ]
    
    lines = text.split('\n')
    education_lines = []
    for line in lines:
        if any(keyword in line.lower() for keyword in education_keywords):
            education_lines.append(line.strip())
    
    # Use spaCy to identify educational organizations
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG" and any(edu in ent.text.lower() for edu in ["university", "college", "institute", "school"]):
            education_lines.append(ent.text)
    
    return list(set(education_lines))  # Remove duplicates

def extract_experience(text):
    """Extract work experience information"""
    # Look for experience sections
    experience_keywords = ["experience", "work experience", "employment", "work history", "professional experience"]
    
    # Try to find experience section
    lines = text.split('\n')
    experience_text = ""
    in_experience_section = False
    
    for line in lines:
        line_lower = line.lower()
        
        if any(keyword in line_lower for keyword in experience_keywords):
            in_experience_section = True
            experience_text += line + "\n"
            continue
        
        # Check if we've reached a new section (typically indicated by a line with few words that might be a header)
        if in_experience_section and line.strip() and len(line.split()) <= 3 and any(c.isupper() for c in line):
            # Check if it's a new non-experience section
            if not any(keyword in line_lower for keyword in experience_keywords):
                if line_lower.strip() and not any(exp in line_lower for exp in experience_keywords):
                    in_experience_section = False
        
        # Add line if we're in the experience section
        if in_experience_section:
            experience_text += line + "\n"
    
    doc = nlp(experience_text)
    organizations = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            organizations.append(ent.text)
    
    # Try to extract job titles using patterns
    job_title_patterns = [
        r'(?:^|\n)([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:at|@|,)\s+([A-Z][a-zA-Z]*)',
        r'(?:^|\n)([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:\(|-).*?(?:\)|-).*?(?:\d{4})'
    ]
    
    job_titles = []
    for pattern in job_title_patterns:
        matches = re.findall(pattern, experience_text)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    job_titles.append(match[0])
                else:
                    job_titles.append(match)
    
    return {
        "experience_text": experience_text,
        "organizations": list(set(organizations)),
        "job_titles": list(set(job_titles))
    }

def parse_resume(file_path):
    """Parse resume file and extract key information"""
    text = extract_text(file_path)
    
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)
    
    # Clean and process text for general analysis
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    
    # Get the most common non-stop words as potential keywords
    word_freq = Counter(filtered_tokens)
    keywords = [word for word, count in word_freq.most_common(20)]
    
    resume_profile = {
        "skills": skills,
        "education": education,
        "experience": experience,
        "keywords": keywords,
        "full_text": text
    }
    
    return resume_profile