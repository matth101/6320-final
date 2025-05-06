import re
import spacy
import os
import json
from nltk.corpus import stopwords
from collections import Counter

import nltk
nltk.data.path.append('C:/Users/MrRic/AppData/Roaming/nltk_data')
nltk.download('punkt_tab', quiet=True)
# Download necessary NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load('en_core_web_sm')
except:
    import sys
    print("You need to download the spaCy model. Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

def extract_skills(text):
    """Extract skills from job description text using NLP techniques"""
    # Common skill keywords - same list as in resume_parser for consistency
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


    skill_variations = {
        "machine learning": ["ml", "machine learning", "deep learning"],
        "artificial intelligence": ["ai", "artificial intelligence"],
        "python": ["python", "py"],
        "javascript": ["javascript", "js", "node.js"],
        "devops": ["devops", "dev ops", "development operations"],
        "backend": ["backend", "back-end", "back end"],
        "frontend": ["frontend", "front-end", "front end"]
    }
    
    soft_skills = [
        "communication", "teamwork", "problem solving", "critical thinking", "creativity",
        "time management", "organization", "adaptability", "flexibility", "leadership",
        "interpersonal", "presentation", "negotiation", "conflict resolution", "decision making",
        "emotional intelligence", "stress management", "work ethic", "attention to detail",
        "customer service", "analytical thinking", "strategic planning", "innovation", "mentoring"
    ]

    # Weight categories
    technical_skills_weight = 1.5
    soft_skills_weight = 1.0
    
    # Combine all skills
    all_skills = technical_skills + soft_skills
    
    text_lower = text.lower()
    
    # Find skills in the text
    found_skills = []
    for skill in all_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    for skill_base, variations in skill_variations.items():
        if any(var in text_lower for var in variations):
            found_skills.append(skill_base)

    # Filter out job titles and generic terms
    exclude_terms = ["engineer", "developer", "specialist", "professional"]
    found_skills = [s for s in found_skills if s.lower() not in exclude_terms]
    
    
    #find additional technical terms
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"] and ent.text.lower() not in found_skills:
            # Check if it might be a technical product/tool/framework
            if any(tech in ent.text.lower() for tech in ["framework", "language", "tool", "platform", "software", "system"]):
                found_skills.append(ent.text.lower())
    
    return found_skills

def extract_education_requirements(text):
    """Extract education requirements from job description"""
    # Education-related keywords and patterns
    education_keywords = [
        "bachelor", "master", "phd", "doctorate", "mba", "bs", "ms", "ba", "ma", "b.s.", "m.s.", 
        "b.a.", "m.a.", "ph.d.", "degree", "university", "college", "graduate"
    ]
    
    # Look for sentences containing education keywords
    education_requirements = []
    
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in education_keywords):
            education_requirements.append(sentence.strip())
    
    return education_requirements

def extract_experience_requirements(text):
    """Extract experience requirements from job description"""
    # Experience-related keywords and patterns
    experience_keywords = [
        "experience", "years", "background", "history", "track record"
    ]
    
    # Look for sentences containing experience keywords
    experience_requirements = []
    
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in experience_keywords):
            experience_requirements.append(sentence.strip())
    
    # Try to extract years of experience using regex patterns
    years_patterns = [
        r'(\d+)[+]?\s+years?\s+(?:of\s+)?experience',
        r'(\d+)[+]?\s+years?\s+(?:of\s+)?(?:work|industry|professional)',
        r'experience\s+(?:of\s+)?(\d+)[+]?\s+years?'
    ]
    
    years_required = None
    for pattern in years_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            years_required = max([int(year) for year in matches])
            break
    
    return {
        "experience_sentences": experience_requirements,
        "years_required": years_required
    }

def extract_job_title(text):
    """Extract job title from job description"""
    # Look for patterns that might indicate job titles
    title_patterns = [
        r'^([A-Z][a-zA-Z\s]+?)(?:[-:]|$)',  # First line with capitalized text
        r'Job Title:?\s*([A-Za-z\s]+)',      # Explicit "Job Title:" format
        r'Position:?\s*([A-Za-z\s]+)',       # Explicit "Position:" format
        r'Role:?\s*([A-Za-z\s]+)'            # Explicit "Role:" format
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
    
    # Fallback to first sentence if no pattern matches
    first_sentence = text.split('.')[0].strip()
    words = first_sentence.split()
    if len(words) <= 5:  # Reasonable length for a job title
        return first_sentence
    
    return "Unknown Job Title"

def extract_company_name(text):
    """Extract company name from job description"""
    company_patterns = [
        r'(?:at|with|for)\s+([A-Z][A-Za-z\s]+?)(?:is|are|has|seeks|\.|,)',
        r'Company:?\s*([A-Za-z\s,\.]+)',
        r'([A-Z][A-Za-z\s]+?)(?:is\s+looking|is\s+seeking|is\s+hiring)'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
    
    doc = nlp(text[:500])  # Use just the first part of the description for efficiency
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    
    if organizations:
        return organizations[0]
    
    return "Unknown Company"

def extract_keywords(text):
    """Extract important keywords from job description"""
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    
    # Get the most common non-stop words as potential keywords
    word_freq = Counter(filtered_tokens)
    keywords = [word for word, count in word_freq.most_common(20)]
    
    return keywords

def parse_job_description(job_text, job_id=None, job_title=None, company=None):
    """Parse job description text and extract key information"""
    # Extract information
    if not job_title:
        job_title = extract_job_title(job_text)
    
    if not company:
        company = extract_company_name(job_text)
    
    skills = extract_skills(job_text)
    education = extract_education_requirements(job_text)
    experience = extract_experience_requirements(job_text)
    keywords = extract_keywords(job_text)
    
    # Create job profile
    job_profile = {
        "job_id": job_id or "job_" + str(hash(job_text) % 10000),
        "job_title": job_title,
        "company": company,
        "skills": skills,
        "education": education,
        "experience": experience,
        "keywords": keywords,
        "full_text": job_text
    }
    
    return job_profile

def load_real_jobs(json_file):
    """Load real job descriptions from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            jobs = data.get('jobs', [])
            return [parse_job_description(
                job['description'], 
                job_id=job['job_id'],
                job_title=job['job_title'],
                company=job['company']
            ) for job in jobs]
    except FileNotFoundError:
        print(f"Warning: {json_file} not found, falling back to sample jobs")
        return load_sample_jobs()

def load_sample_jobs(directory="data/jobs"):
    """Load sample job descriptions from files in the specified directory"""
    jobs = []
    
    os.makedirs(directory, exist_ok=True)
    
    # Check if directory is empty (no job files)
    if not os.listdir(directory):
        # Add sample job descriptions programmatically
        sample_jobs = [
            {
                "job_id": "job_001",
                "job_title": "Data Scientist",
                "company": "Tech Innovations Inc.",
                "description": """
                Data Scientist at Tech Innovations Inc.
                
                We are looking for a talented Data Scientist to join our growing team. The ideal candidate will have strong skills in machine learning, statistics, and data analysis.
                
                Requirements:
                - Bachelor's degree in Computer Science, Statistics, or related field. Master's degree preferred.
                - 3+ years of experience in data science or related role
                - Proficiency in Python, R, or similar programming languages
                - Experience with machine learning frameworks such as TensorFlow, PyTorch, or scikit-learn
                - Strong understanding of statistical analysis and modeling
                - Experience with data visualization tools (Tableau, PowerBI, matplotlib)
                - Knowledge of SQL and database systems
                - Excellent problem-solving and communication skills
                
                Responsibilities:
                - Develop machine learning models to solve business problems
                - Analyze large datasets to extract meaningful insights
                - Create visualizations and dashboards to communicate findings
                - Collaborate with cross-functional teams to implement data-driven solutions
                - Stay up-to-date with the latest developments in data science and machine learning
                
                If you are passionate about using data to drive decision-making and want to be part of an innovative team, we'd love to hear from you!
                """
            },
            {
                "job_id": "job_002",
                "job_title": "Software Engineer",
                "company": "Global Solutions Ltd.",
                "description": """
                Software Engineer at Global Solutions Ltd.
                
                We are seeking a talented Software Engineer to join our development team. The successful candidate will help design, develop, and maintain high-quality software solutions.
                
                Requirements:
                - Bachelor's degree in Computer Science or related field
                - 2+ years of experience in software development
                - Strong proficiency in Java, C#, or Python
                - Experience with web frameworks (Spring, Django, ASP.NET)
                - Knowledge of frontend technologies (HTML, CSS, JavaScript)
                - Familiarity with database systems (SQL, NoSQL)
                - Understanding of software development methodologies (Agile, Scrum)
                - Good problem-solving and communication skills
                
                Responsibilities:
                - Design and develop software applications according to requirements
                - Write clean, efficient, and well-documented code
                - Debug and resolve software defects
                - Participate in code reviews and contribute to team best practices
                - Collaborate with cross-functional teams to deliver solutions
                - Stay current with emerging technologies and trends
                
                If you are passionate about building high-quality software and want to work in a collaborative environment, we want to hear from you!
                """
            },
            {
                "job_id": "job_003",
                "job_title": "Machine Learning Engineer",
                "company": "AI Futures",
                "description": """
                Machine Learning Engineer at AI Futures
                
                AI Futures is looking for a skilled Machine Learning Engineer to help build and deploy ML models that power our next-generation AI products.
                
                Requirements:
                - Master's degree or PhD in Computer Science, Machine Learning, or related field
                - 4+ years of experience in machine learning engineering
                - Strong programming skills in Python and familiarity with ML frameworks (TensorFlow, PyTorch)
                - Experience with NLP, computer vision, or reinforcement learning
                - Knowledge of cloud platforms (AWS, GCP, Azure) and ML deployment
                - Understanding of data structures, algorithms, and software design
                - Excellent problem-solving and analytical thinking skills
                
                Responsibilities:
                - Design and implement machine learning models for various applications
                - Optimize existing models for performance and scalability
                - Deploy ML solutions to production environments
                - Collaborate with data scientists and engineers to integrate ML into products
                - Research and implement state-of-the-art ML techniques
                - Monitor and maintain ML systems in production
                
                Join our team and help shape the future of AI technology!
                """
            },
            {
                "job_id": "job_004",
                "job_title": "Frontend Developer",
                "company": "WebCraft Solutions",
                "description": """
                Frontend Developer at WebCraft Solutions
                
                We are hiring a Frontend Developer to create engaging and responsive user interfaces for our web applications.
                
                Requirements:
                - Bachelor's degree in Computer Science or related field (or equivalent experience)
                - 2+ years of experience in frontend development
                - Strong proficiency in HTML, CSS, and JavaScript
                - Experience with modern frontend frameworks (React, Angular, or Vue.js)
                - Knowledge of responsive design and cross-browser compatibility
                - Familiarity with UI/UX principles and design tools
                - Basic understanding of backend integration
                - Good problem-solving skills and attention to detail
                
                Responsibilities:
                - Develop responsive and interactive user interfaces
                - Implement UI components using modern frameworks
                - Ensure cross-browser compatibility and performance
                - Collaborate with designers and backend developers
                - Optimize applications for maximum speed and scalability
                - Stay updated with emerging frontend technologies
                
                If you have a passion for creating beautiful and functional web interfaces, apply now!
                """
            },
            {
                "job_id": "job_005",
                "job_title": "Data Engineer",
                "company": "DataFlow Systems",
                "description": """
                Data Engineer at DataFlow Systems
                
                DataFlow Systems is seeking a skilled Data Engineer to design and implement data pipelines and infrastructure.
                
                Requirements:
                - Bachelor's degree in Computer Science, Engineering, or related field
                - 3+ years of experience in data engineering or related role
                - Strong programming skills in Python, Java, or Scala
                - Experience with data processing frameworks (Spark, Hadoop)
                - Knowledge of SQL and NoSQL databases
                - Familiarity with ETL tools and data warehousing concepts
                - Experience with cloud platforms (AWS, GCP, Azure)
                - Good understanding of data modeling and architecture
                
                Responsibilities:
                - Design and implement data pipelines for efficient data processing
                - Build and maintain data infrastructure and architecture
                - Optimize data flows for performance and reliability
                - Collaborate with data scientists and analysts to support their data needs
                - Implement data quality measures and monitoring
                - Document data systems and processes
                
                Join our team and help build the data infrastructure of tomorrow!
                """
            }
        ]
        
        # Save sample jobs to files
        for job in sample_jobs:
            job_file = os.path.join(directory, f"{job['job_id']}.json")
            with open(job_file, 'w') as f:
                json.dump(job, f, indent=2)
    
    # Load all job files from directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                job_data = json.load(f)
                job_profile = parse_job_description(
                    job_data['description'], 
                    job_id=job_data.get('job_id'),
                    job_title=job_data.get('job_title'),
                    company=job_data.get('company')
                )
                jobs.append(job_profile)
        elif filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                job_text = f.read()
                job_id = os.path.splitext(filename)[0]
                job_profile = parse_job_description(job_text, job_id=job_id)
                jobs.append(job_profile)
    
    return jobs