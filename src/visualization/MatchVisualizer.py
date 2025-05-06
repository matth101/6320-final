import matplotlib.pyplot as plt
import seaborn as sns

class MatchVisualizer:
    # def __init__(self):
        # Set style for better-looking plots
        # plt.style.use('seaborn')
        
    def create_match_summary(self, match_results, save_path=None):
        """Create a summary visualization of job matches"""
        # Extract data for visualization
        companies = [result['company'] for result in match_results]
        overall_scores = [result['overall_match'] * 100 for result in match_results]
        skill_scores = [result['skill_match'] * 100 for result in match_results]
        semantic_scores = [result['semantic_similarity'] * 100 for result in match_results]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot overall match scores
        sns.barplot(x=overall_scores, y=companies, ax=ax1, color='skyblue')
        ax1.set_title('Overall Match Scores by Company')
        ax1.set_xlabel('Match Score (%)')
        ax1.set_ylabel('Company')

        # Plot detailed scores for top match
        top_match = match_results[0]  # Assuming results are sorted
        categories = ['Overall', 'Skills', 'Semantic', 'Keywords']
        scores = [
            top_match['overall_match'] * 100,
            top_match['skill_match'] * 100,
            top_match['semantic_similarity'] * 100,
            top_match['keyword_match'] * 100
        ]

        sns.barplot(x=categories, y=scores, ax=ax2, palette='viridis')
        ax2.set_title(f'Detailed Scores for Top Match: {top_match["company"]}')
        ax2.set_ylabel('Score (%)')
        ax2.set_ylim(0, 100)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def create_skills_comparison(self, match_result, save_path=None):
        """Create a visualization of matching vs missing skills"""
        matching_skills = match_result['matching_skills'][:5]  # Top 5 matching skills
        missing_skills = match_result['missing_skills'][:5]    # Top 5 missing skills

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot matching skills in green
        ax.barh(range(len(matching_skills)), 
                [1] * len(matching_skills), 
                label='Matching Skills', 
                color='green', 
                alpha=0.6)

        # Plot missing skills in red
        ax.barh(range(len(matching_skills), len(matching_skills) + len(missing_skills)), 
                [1] * len(missing_skills), 
                label='Missing Skills', 
                color='red', 
                alpha=0.6)

        # Customize the plot
        ax.set_yticks(range(len(matching_skills) + len(missing_skills)))
        ax.set_yticklabels(matching_skills + missing_skills)
        ax.set_title(f'Skills Analysis for {match_result["company"]}')
        ax.set_xlabel('Skills Status')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()