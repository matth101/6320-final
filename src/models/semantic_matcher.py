import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re
# from ..visualization.viz import MatchVisualizer

nltk.data.path.append('C:/Users/MrRic/AppData/Roaming/nltk_data')
nltk.download('punkt_tab', quiet=True)
# Download necessary NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)

class ResumeJobMatcher:
    def __init__(self):
        # Load the sentence transformer model for semantic matching
        try:
            self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except:
            # Fallback to TF-IDF if model loading fails
            self.semantic_model = None
            print("Warning: Sentence transformer model not available, falling back to TF-IDF.")
        
        # Initialize TF-IDF vectorizer as backup or for keyword matching
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Set up stop words for text cleaning
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        if not text:
            return ""
            
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def calculate_skill_match(self, resume_skills, job_skills):
        """Calculate the match score based on skills"""
        if not resume_skills or not job_skills:
            return 0.0
            
        resume_skills_lower = [s.lower() for s in resume_skills]
        job_skills_lower = [s.lower() for s in job_skills]
        
        matching_skills = set(resume_skills_lower).intersection(set(job_skills_lower))
        
        # Calculate match percentage
        if len(job_skills_lower) == 0:
            return 0.0
        
        match_percentage = len(matching_skills) / len(job_skills_lower)
        
        missing_skills = set(job_skills_lower) - set(resume_skills_lower)
        
        return match_percentage, list(matching_skills), list(missing_skills)
    
    def calculate_semantic_similarity(self, resume_text, job_text):
        """Calculate semantic similarity between resume and job description using transformers"""
        if not resume_text or not job_text:
            return 0.0
            
        resume_text_clean = self.preprocess_text(resume_text)
        job_text_clean = self.preprocess_text(job_text)
        
        if not resume_text_clean or not job_text_clean:
            return 0.0
            
        # Use sentence transformer model if available
        if self.semantic_model:
            # Get embeddings
            resume_embedding = self.semantic_model.encode([resume_text_clean])[0]
            job_embedding = self.semantic_model.encode([job_text_clean])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            return similarity
        
        # Fallback to TF-IDF
        try:
            tfidf_matrix = self.tfidf.fit_transform([resume_text_clean, job_text_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def calculate_keyword_match(self, resume_keywords, job_keywords):
        """Calculate match based on keywords"""
        if not resume_keywords or not job_keywords:
            return 0.0
            
        resume_keywords_lower = [k.lower() for k in resume_keywords]
        job_keywords_lower = [k.lower() for k in job_keywords]
        
        matching_keywords = set(resume_keywords_lower).intersection(set(job_keywords_lower))
        
        # Calculate match percentage
        if len(job_keywords_lower) == 0:
            return 0.0
        
        match_percentage = len(matching_keywords) / len(job_keywords_lower)
        return match_percentage
    
    def identify_education_gap(self, resume_education, job_education):
        """Identify gaps between resume education and job requirements"""
        if not job_education:
            return "No specific education requirements identified"
        
        # Check if resume has any education listed
        if not resume_education:
            return "Resume does not list education details, but job requires education"
        
        # Basic pattern matching for degree levels
        degree_levels = {
            "bachelor": 1,
            "bs": 1,
            "ba": 1,
            "b.s.": 1,
            "b.a.": 1,
            "undergraduate": 1,
            "master": 2,
            "ms": 2,
            "ma": 2,
            "m.s.": 2,
            "m.a.": 2,
            "mba": 2,
            "graduate": 2,
            "phd": 3,
            "ph.d.": 3,
            "doctorate": 3,
            "doctoral": 3
        }
        
        job_edu_level = 0
        job_edu_text = " ".join(job_education).lower() if isinstance(job_education, list) else job_education.lower()
        
        for degree, level in degree_levels.items():
            if degree in job_edu_text:
                job_edu_level = max(job_edu_level, level)
        
        resume_edu_level = 0
        resume_edu_text = " ".join(resume_education).lower() if isinstance(resume_education, list) else resume_education.lower()
        
        for degree, level in degree_levels.items():
            if degree in resume_edu_text:
                resume_edu_level = max(resume_edu_level, level)
        
        if job_edu_level == 0:
            return "No specific degree level requirement identified"
        
        if resume_edu_level == 0:
            return "Education level not clearly identified in resume"
        
        if resume_edu_level < job_edu_level:
            if job_edu_level == 1:
                return "Job requires Bachelor's degree, potential education gap"
            elif job_edu_level == 2:
                return "Job requires Master's degree, potential education gap"
            elif job_edu_level == 3:
                return "Job requires PhD/Doctorate, potential education gap"
        
        return "Education requirements appear to be met"
    
    def identify_experience_gap(self, resume_experience, job_experience):
        """Identify gaps between resume experience and job requirements"""
        # Extract years required from job if available
        years_required = job_experience.get('years_required', None)
        
        # If no specific years required, return a general statement
        if not years_required:
            return "No specific years of experience requirement identified"
        
        # Try to infer years of experience from resume
        # This is a simplistic approachh.
        resume_exp_text = resume_experience.get('experience_text', '')
        
        # Look for patterns indicating years of experience
        year_patterns = [
            r'(\d+)[+]?\s+years?\s+(?:of\s+)?experience',
            r'(\d+)[+]?\s+years?\s+(?:of\s+)?(?:work|industry|professional)',
            r'experience\s+(?:of\s+)?(\d+)[+]?\s+years?'
        ]
        
        resume_years = None
        for pattern in year_patterns:
            matches = re.findall(pattern, resume_exp_text.lower())
            if matches:
                resume_years = max([int(year) for year in matches])
                break
        
        if resume_years is None:
            # Count number of job roles/positions as a rough estimate
            org_count = len(resume_experience.get('organizations', []))
            title_count = len(resume_experience.get('job_titles', []))
            
            if org_count > 0 or title_count > 0:
                # Rough heuristic: assume 2 years per position if multiple positions
                if max(org_count, title_count) > 1:
                    resume_years = max(org_count, title_count) * 2
                else:
                    resume_years = 1  # At least some experience
            else:
                return f"Job requires {years_required}+ years of experience, resume experience unclear"
        
        if resume_years < years_required:
            gap = years_required - resume_years
            return f"Job requires {years_required}+ years of experience, potential gap of {gap} years"
        
        return "Experience requirements appear to be met"
    
    def match_resume_to_job(self, resume_profile, job_profile):
        """Match a resume to a job and return detailed match information"""
        skill_match, matching_skills, missing_skills = self.calculate_skill_match(
            resume_profile.get('skills', []),
            job_profile.get('skills', [])
        )
        
        # Calculate semantic similarity between full texts
        semantic_similarity = self.calculate_semantic_similarity(
            resume_profile.get('full_text', ''),
            job_profile.get('full_text', '')
        )
        
        keyword_match = self.calculate_keyword_match(
            resume_profile.get('keywords', []),
            job_profile.get('keywords', [])
        )
        
        # Adjust weights to emphasize technical matches
        overall_match = (
            skill_match * 0.6 +          # Increased from 0.5
            semantic_similarity * 0.3 +   # Same
            keyword_match * 0.1          # Decreased from 0.2
        )
        education_gap = self.identify_education_gap(
            resume_profile.get('education', []),
            job_profile.get('education', {})
        )
        
        experience_gap = self.identify_experience_gap(
            resume_profile.get('experience', {}),
            job_profile.get('experience', {})
        )
        
        # Return match results
        match_results = {
            'job_id': job_profile.get('job_id', ''),
            'job_title': job_profile.get('job_title', 'Unknown Job'),
            'company': job_profile.get('company', 'Unknown Company'),
            'overall_match': overall_match,
            'skill_match': skill_match,
            'semantic_similarity': semantic_similarity,
            'keyword_match': keyword_match,
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'education_gap': education_gap,
            'experience_gap': experience_gap
        }
        
        return match_results
    
    def match_resume_to_jobs(self, resume_profile, job_profiles):
        """Match a resume to multiple jobs and return sorted results"""
        match_results = []
        
        for job_profile in job_profiles:
            result = self.match_resume_to_job(resume_profile, job_profile)
            match_results.append(result)
        
        # Sort by overall match score, descending
        match_results.sort(key=lambda x: x['overall_match'], reverse=True)
        
        return match_results