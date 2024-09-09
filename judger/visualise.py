import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#####################################################################
filenames = ['results/financial_revised_ques_financial_gpt_4o_mini.json', 
             'results/financial_revised_ques_financial_individual_gpt_4o_mini.json', 
             'results/financial_revised_ques_financial_individual_deepseek.json']
labels = ['GPT4o_combined', 'GPT4o_individual', 'Deepseek_individual']
#####################################################################

def main():
    num_files = len(filenames)
    if num_files > 3:
        print("WARNING: Too many files may result in poor visualisation!")
    plt.figure(figsize=(12, 3*num_files))
    for i in range(num_files):
        file = filenames[i]
        label = labels[i]
        plt.subplot(num_files, 1, i+1)
        visualize_single_file(file, label)
    plt.tight_layout()
    plt.show()

def visualize_single_file(filename, label):
    with open(filename, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data).T
    df['index'] = df.index
    
    melted_data = df.melt(id_vars=["index"], 
                          value_vars=["computational_complexity", "data_integration_needs", "business_understanding_difficulty"], 
                          var_name="metric", 
                          value_name="level")

    sns.scatterplot(data=melted_data, x="index", y="level", style="metric", s=60, 
                    markers={"computational_complexity": "o",    # Circle (hollow)
                             "data_integration_needs": "X",       # X
                             "business_understanding_difficulty": "s"},  # Square (hollow)
                    edgecolor="black", facecolors='none')  # Hollow markers

    plt.title(f"Metric Levels for {label}")
    plt.xlabel("Question Index")
    plt.ylabel("Metric Level (0-4)")
    plt.xticks(rotation=45)
    plt.yticks([0, 1, 2, 3, 4])  
    plt.legend(loc='upper right')

if __name__ == "__main__":
    main()