# analyze_titanic.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
file_path = 'C:\\Users\\Ali\\Downloads\\titanic\\train.csv'
titanic_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(titanic_data.head())

# Data Cleaning and Filtering
titanic_data = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic_data = titanic_data.dropna()  # Handle missing values (you might want to handle missing values more carefully)
titanic_data = titanic_data[titanic_data['Age'] > 18]  # Filter specific rows (e.g., passengers with age greater than 18)

# Data Visualization using Matplotlib and Seaborn

# Set the style for Seaborn
sns.set(style="whitegrid")

# Visualize the distribution of ages using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Create a KDE plot for the 'Age' column
plt.figure(figsize=(10, 6))
sns.kdeplot(titanic_data['Age'], fill=True, color='green', label='Age Distribution')
plt.title('Kernel Density Estimate (KDE) Plot for Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# Create a count plot for the 'Survived' column
sns.countplot(x='Survived', hue='Survived', data=titanic_data, palette='viridis', legend=False)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Create a bar plot for the 'Pclass' column
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Fare', data=titanic_data, palette='muted')
plt.title('Average Fare by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Average Fare')
plt.show()

# Create a box plot for the 'Fare' column
plt.figure(figsize=(10, 6))
sns.boxplot(x='Embarked', y='Fare', data=titanic_data, palette='pastel')
plt.title('Fare Distribution by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Fare')
plt.show()
