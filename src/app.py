import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pathlib import Path
from parsers.resume_parser import parse_resume
from parsers.job_parser import load_sample_jobs, load_real_jobs
from models.semantic_matcher import ResumeJobMatcher
import json

from visualization.MatchVisualizer import MatchVisualizer

import nltk
nltk.data.path.append('C:/Users/MrRic/AppData/Roaming/nltk_data')
nltk.download('punkt_tab', quiet=True) 

def setup_directories():
    directories = [
        'data/resumes',
        'data/job_descriptions',
        'notebooks'
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def display_match_results(results):
    best_match = max(result['overall_match'] for result in results) * 100
    grade_letter = 'A' if best_match >= 75 else 'B' if best_match >= 65 else 'C' if best_match >= 55 else 'D' if best_match >= 45 else 'F'
    
    print("\n=== Resume Grading Report ===")
    print(f"\nOverall Resume Grade: {grade_letter} ({best_match:.1f}%)")
    print("Based on strongest job match and key factors:")
    print("- Technical skill alignment")
    print("- Experience relevance")
    print("- Keyword optimization")

    print("\n=== Job Match Results ===")
    print("\nScore Explanations:")
    print("- Overall Match: Weighted combination of all factors below")
    print("- Skills Match: Direct matches of technical and soft skills")
    print("- Semantic Similarity: How well your resume content aligns with job requirements")
    print("- Keyword Match: Specific keyword overlap between resume and job posting")
    print("\nDetailed Results:")
    
    for idx, result in enumerate(results, 1):
        print(f"\n{idx}. {result['job_title']} at {result['company']}")
        print(f"   Overall Match: {result['overall_match']*100:.1f}%")
        print(f"   - Skills Match: {result['skill_match']*100:.1f}%")
        print(f"   - Semantic Similarity: {result['semantic_similarity']*100:.1f}%")
        print(f"   - Keyword Match: {result['keyword_match']*100:.1f}%")
        
        print("\n   Matching Skills:")
        for skill in result['matching_skills'][:5]:  # Show top 5 matching skills
            print(f"   ✓ {skill}")
            
        if result['missing_skills']:
            print("\n   Missing Skills:")
            for skill in result['missing_skills'][:5]:  # Show top 5 missing skills
                print(f"   × {skill}")
        
        print(f"\n   Education: {result['education_gap']}")
        print(f"   Experience: {result['experience_gap']}")
        print("\n" + "-"*50)

def main():
    parser = argparse.ArgumentParser(description='Resume-Job Matcher')
    parser.add_argument('resume_path', help='Path to the resume file (PDF or DOCX)')
    parser.add_argument('--save', action='store_true', help='Save results to JSON file')
    args = parser.parse_args()

    setup_directories()

    try:
        print("Parsing resume...")
        resume_profile = parse_resume(args.resume_path)

        print("Loading job descriptions...")
        # job_profiles = load_sample_jobs()
        job_profiles = load_real_jobs()

        matcher = ResumeJobMatcher()

        print("Performing matching analysis...")
        match_results = matcher.match_resume_to_jobs(resume_profile, job_profiles)

        display_match_results(match_results)

        if args.save:
            output_file = 'match_results.json'
            with open(output_file, 'w') as f:
                json.dump(match_results, f, indent=2)
            print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    visualizer = MatchVisualizer()
    visualizer.create_match_summary(match_results, 'resume_analysis.png')
    visualizer.create_skills_comparison(match_results[0], 'skills_breakdown.png')
    print("\nVisualizations saved as 'resume_analysis.png' and 'skills_breakdown.png'")

    return 0

if __name__ == "__main__":
    exit(main())