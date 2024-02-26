import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_file = 'combined_results.csv'
df = pd.read_csv(data_file)

# List of output metrics to plot
metrics = ['train_loss', 'val_loss']

for metric in metrics:
    # Pivot the DataFrame to create a matrix suitable for a heatmap
    heatmap_data = df.pivot(index='alpha', columns='beta', values=metric)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    
    # Title and labels
    plt.title(f"Heatmap of {metric} for Alpha vs Beta")
    plt.xlabel("Beta")
    plt.ylabel("Alpha")
    
    # Save the heatmap as a .png file
    plt.savefig(f"{metric}_heatmap.png")
    plt.close()
