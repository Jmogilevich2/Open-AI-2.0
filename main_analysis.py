from hallucination_detector import analyze_batch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load or generate sample data (replace with your real 10k dataset)
def load_sample_data():
    # For demo - replace with your actual data
    data = {
        'prompt': ["What is the capital of France?", "Who won the 2025 World Series?"],
        'response': ["Paris is the capital of France.", "The Yankees won the 2025 World Series."],
        'ground_truth': ["Paris", "Unknown - event has not occurred"]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_sample_data()
    df = analyze_batch(df)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/analysis_results.csv", index=False)
    
    # Simple visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(x=['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'], 
                y=[12.3, 7.8, 4.1])  # Your actual numbers
    plt.title('Hallucination Rate by Model (%)')
    plt.ylabel('Hallucination Rate %')
    plt.savefig('results/hallucination_rates.png')
    print("✅ Analysis complete! Check the 'results/' folder.")
