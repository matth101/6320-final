# Resume Grader: NLP-Based ATS Analysis Tool

CS 6320 

View demo: https://youtu.be/Zj6z0MO5rBQ

## Overview
Resume Grader is a Python-based tool that analyzes resumes against job descriptions using Natural Language Processing (NLP) techniques. I wanted to simulate at a basic level how Applicant Tracking Systems (ATS) evaluate resumes, providing detailed matching scores and actionable feedback.

## Features
- Resume parsing (PDF/DOCX support)
- Semantic matching using BERT-based transformers
- Comprehensive skills detection and analysis
- Education and experience requirement matching
- Visual match analysis and skills gap identification
- Real-time job description analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-grader.git
cd resume-grader

# Install required packages
pip install -r requirements.txt

# Download required NLTK data
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Dependencies
- Python 3.8+
- PyPDF2
- python-docx
- spaCy
- NLTK
- sentence-transformers
- matplotlib
- seaborn
- numpy
- pandas

## Project Structure

resume-grader/
├── data/
│ ├── resumes/ # Place resumes here
│ └── job_descriptions/ # Job description data
├── src/
│ ├── parsers/ # Resume and job parsing
│ ├── models/ # Matching algorithms
│ └── visualization/ # Data visualization
├── notebooks/ # Demo notebooks
└── requirements.txt