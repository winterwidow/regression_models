import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("KO_1919-09-06_2025-04-17.csv")  

# Select features of interest
cols = ["low", "high", "volume"]
correlation_matrix = df[cols].corr()

# Visualize
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Low, High, Volume)")
plt.show()