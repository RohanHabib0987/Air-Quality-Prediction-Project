# Air Quality Prediction Project

This repository contains a Jupyter Notebook (or Python script) that demonstrates a machine learning workflow for predicting air quality. The process covers data loading, exploratory data analysis (EDA), preprocessing, and the application of several classification algorithms with their evaluation metrics.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Mount Google Drive](#1-mount-google-drive)
  - [2. Load Dataset](#2-load-dataset)
  - [3. Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [4. Data Preprocessing](#4-data-preprocessing)
    - [Outlier Handling](#outlier-handling)
    - [Missing Data Handling](#missing-data-handling)
    - [Data Normalization](#data-normalization)
    - [Categorical Encoding](#categorical-encoding)
    - [Skewed Data Transformation](#skewed-data-transformation)
  - [5. Model Training and Evaluation](#5-model-training-and-evaluation)
    - [Classification Algorithms](#classification-algorithms)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Visualizing Results](#visualizing-results)
  - [6. ROC-AUC Analysis](#6-roc-auc-analysis)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The main objective of this project is to build and evaluate machine learning models to predict air quality based on various environmental features. The project follows a standard machine learning pipeline:

1.  **Data Loading**: Importing the dataset from Google Drive.
2.  **Exploratory Data Analysis (EDA)**: Understanding the data's characteristics, distributions, and relationships.
3.  **Data Preprocessing**: Handling missing values, outliers, normalizing features, encoding categorical variables, and transforming skewed data.
4.  **Model Training**: Applying Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN) algorithms.
5.  **Model Evaluation**: Assessing model performance using accuracy, precision, recall, F1-score, and ROC-AUC.

---

## Dataset

The dataset used for this project is `AIR-QUALITY.zip` which is expected to contain a CSV file within it. Please ensure this file is accessible in your Google Drive at the specified path.

---

## Prerequisites

Before running the code, ensure you have the following installed:

-   Python 3.x
-   Jupyter Notebook (recommended for interactive execution)
-   Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

---

## Usage

The following steps outline how to run the code and understand each part of the machine learning pipeline.

### 1. Mount Google Drive

If you're running this in Google Colab, you'll need to mount your Google Drive to access the dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
import pandas as pd

file_path = '/content/drive/MyDrive/AIR POLUTION/AIR-QUALITY.zip'

try:
  df = pd.read_csv(file_path)
  print("Dataset loaded successfully!")
  print(df.head()) # Display the first few rows of the dataframe
except FileNotFoundError:
  print(f"Error: File not found at {file_path}. Please check the file path.")
except Exception as e:
  print(f"An error occurred: {e}")
```


## . Exploratory Data Analysis (EDA)
This section provides a high-level view of the dataset, including descriptive statistics, data types, unique values, and missing data.

```python

# Display basic descriptive statistics for each numerical variable
print(df.describe())

# Display descriptive statistics for all variables (including categorical)
print(df.describe(include='all'))

# Get the data types of each column
print(df.dtypes)

# Get the number of unique values in each column
print(df.nunique())

# Identify missing values
print(df.isnull().sum())

# Summarize categorical variables with counts and proportions
for col in df.select_dtypes(include=['object', 'category']):
  print(f"Summary for '{col}':")
  print(df[col].value_counts())
  print(df[col].value_counts(normalize=True) * 100)
  print("-" * 20)

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize data distributions with histograms
for column in df.select_dtypes(include=['number']):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Visualize data distributions with boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.select_dtypes(include=['number'])):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
plt.show()

# Use scatter plots to examine relationships between variables
numerical_cols = df.select_dtypes(include=['number']).columns
for i in range(len(numerical_cols)):
    for j in range(i + 1, len(numerical_cols)):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[numerical_cols[i]], y=df[numerical_cols[j]])
        plt.title(f'Scatter Plot: {numerical_cols[i]} vs. {numerical_cols[j]}')
        plt.xlabel(numerical_cols[i])
        plt.ylabel(numerical_cols[j])
        plt.show()

# Use heatmaps to examine relationships between variables
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Plot bar charts for categorical data distribution
for col in df.select_dtypes(include=['object', 'category']):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Bar Chart of {col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
```

## 4. Data Preprocessing
This crucial step prepares the data for machine learning models by handling imperfections and transforming features.

Outlier Handling
Outliers are removed using the Interquartile Range (IQR) method for a specified column.
```python

import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Example usage: Replace 'CO' with your actual numerical column name
df_no_outliers = remove_outliers_iqr(df, 'CO')
Missing Data Handling
Various strategies for handling missing values are demonstrated, including dropping rows, filling with mean/median/mode, forward/backward fill, and imputation using SimpleImputer. It's important to choose the most appropriate method based on your dataset and domain knowledge.

##python

# Method 1: Removing rows with missing values
df_dropped = df.dropna()

# Method 2: Filling missing values with the mean (for numerical columns)
numerical_cols = df.select_dtypes(include=['number']).columns
df_filled_mean = df.copy()
for col in numerical_cols:
    df_filled_mean[col] = df_filled_mean[col].fillna(df_filled_mean[col].mean())

# Method 3: Filling missing values with the median (for numerical columns)
df_filled_median = df.copy()
for col in numerical_cols:
    df_filled_median[col] = df_filled_median[col].fillna(df_filled_median[col].median())

# Method 4: Filling missing values with a specific value (e.g., 0)
df_filled_zero = df.fillna(0)

# Method 5: Filling missing values with the most frequent value (mode) for categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_filled_mode = df.copy()
for col in categorical_cols:
    df_filled_mode[col] = df_filled_mode[col].fillna(df_filled_mode[col].mode()[0])

# Method 6: Filling missing values using forward fill or backward fill
df_ffill = df.fillna(method='ffill')
df_bfill = df.fillna(method='bfill')

# Method 7: Imputation using scikit-learn's SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = df.copy()
df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Display the head of DataFrames after each method to compare
print("Original DataFrame (head):\n", df.head())
print("\nDataFrame with rows containing missing values dropped (head):\n", df_dropped.head())
print("\nDataFrame with missing numerical values filled with the mean (head):\n", df_filled_mean.head())
print("\nDataFrame with missing numerical values filled with the median (head):\n", df_filled_median.head())
print("\nDataFrame with missing values filled with 0 (head):\n", df_filled_zero.head())
print("\nDataFrame with missing categorical values filled with the mode (head):\n", df_filled_mode.head())
print("\nDataFrame with missing values filled using forward fill (head):\n", df_ffill.head())
print("\nDataFrame with missing values filled using backward fill (head):\n", df_bfill.head())
print("\nDataFrame with missing numerical values imputed using SimpleImputer (head):\n", df_imputed.head())
Data Normalization
Numerical features are scaled to a common range (0 to 1) using MinMaxScaler to prevent features with larger values from dominating the model.

Python

from sklearn.preprocessing import MinMaxScaler

numerical_cols = df_imputed.select_dtypes(include=['number']).columns
scaler = MinMaxScaler()
df_normalized = df_imputed.copy()
df_normalized[numerical_cols] = scaler.fit_transform(df_imputed[numerical_cols])
print(df_normalized.head())
Categorical Encoding
Categorical variables are converted into numerical representations using LabelEncoder. This is necessary for most machine learning algorithms.

Python

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for column in df_normalized.select_dtypes(include=['object', 'category']):
    df_normalized[column] = label_encoder.fit_transform(df_normalized[column])
print(df_normalized.head())
Skewed Data Transformation
For numerical features with skewed distributions, PowerTransformer (Yeo-Johnson) is applied to make them more Gaussian-like, which can improve model performance.

Python

from sklearn.preprocessing import PowerTransformer

numerical_cols = df_normalized.select_dtypes(include=['number']).columns
pt = PowerTransformer(method='yeo-johnson')
df_transformed = df_normalized.copy()
df_transformed[numerical_cols] = pt.fit_transform(df_normalized[numerical_cols])
print(df_transformed.head())
```
## 5. Model Training and Evaluation
This section focuses on training different classification models and evaluating their performance.

First, define your features (X) and target variable (y), then split the data into training and testing sets.
```python
Python

from sklearn.model_selection import train_test_split

# Replace 'Air Quality' with your actual target variable column name
X = df_normalized.drop('Air Quality', axis=1) # Use df_normalized or df_transformed based on your last processing step
y = df_normalized['Air Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Classification Algorithms
Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN) are applied.

Python

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    return accuracy, y_pred # Return accuracy and predictions for further metrics

# Evaluate each model and store accuracies and predictions
accuracy_logreg, y_pred_logreg = evaluate_model(LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression")
accuracy_dt, y_pred_dt = evaluate_model(DecisionTreeClassifier(random_state=42), "Decision Tree")
accuracy_knn, y_pred_knn = evaluate_model(KNeighborsClassifier(n_neighbors=5), "KNN")
Evaluation Metrics
Accuracy, Precision, Recall, and F1-score are calculated and visualized for each model.

Python

from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# --- Logistic Regression Specific Metrics (from earlier prompt) ---
# Assuming y_pred_logreg is available from the evaluate_model function call
Precision1 = precision_score(y_test, y_pred_logreg, average='weighted')
Recall1 = recall_score(y_test, y_pred_logreg, average='weighted')
F1_score1 = f1_score(y_test, y_pred_logreg, average='weighted')

print(f"Precision1: {Precision1}")
print(f"Recall1: {Recall1}")
print(f"F1 score1: {F1_score1}")

results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Value': [Precision1, Recall1, F1_score1]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Value', data=results_df)
plt.title('Logistic Regression Performance Metrics')
plt.ylabel('Score')
plt.show()

# --- General Recall Comparison for all three algorithms ---
recall_logreg = recall_score(y_test, y_pred_logreg, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')

print(f"Logistic Regression Recall: {recall_logreg}")
print(f"Decision Tree Recall: {recall_dt}")
print(f"KNN Recall: {recall_knn}")

algorithms = ['Logistic Regression', 'Decision Tree', 'KNN']
recalls = [recall_logreg, recall_dt, recall_knn]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, recalls, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.xlabel("Algorithms")
plt.ylabel("Recall")
plt.title("Comparison of Algorithm Recalls")
plt.ylim(0, 1)
plt.show()

# --- F1 Score for a specific model (e.g., Logistic Regression) ---
# If you want to show F1 for each model, you'd calculate it for y_pred_dt and y_pred_knn too
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted')
print(f"F1 Score for Logistic Regression: {f1_logreg}")

plt.figure(figsize=(8, 6))
plt.bar(['Logistic Regression'], [f1_logreg], color=['skyblue'])
plt.xlabel("Algorithms")
plt.ylabel("F1 Score")
plt.title("F1 Score for Logistic Regression")
plt.ylim(0, 1)
plt.show()
Visualizing Results
A bar graph comparing the accuracy of all three algorithms is presented.

Python

import matplotlib.pyplot as plt

algorithms = ['Logistic Regression', 'Decision Tree', 'KNN']
accuracies = [accuracy_logreg, accuracy_dt, accuracy_knn]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Comparison of Algorithm Accuracies")
plt.ylim(0, 1)
plt.show()
```
## 6. ROC-AUC Analysis
The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are generated to assess the models' ability to distinguish between classes.
```python
Python

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Assuming 'classifier' is your chosen model (e.g., RandomForestClassifier or LogisticRegression)
# Ensure your chosen model has a 'predict_proba' method for ROC-AUC.
# For Logistic Regression, use logreg_model.predict_proba(X_test)
# For Decision Tree, use dt_classifier.predict_proba(X_test)
# For KNN, use knn_classifier.predict_proba(X_test)

# Example using the Logistic Regression model (replace with the model you want to analyze)
try:
    y_pred_proba = logreg_model.predict_proba(X_test)

    # For multi-class classification, ROC AUC is often calculated One-vs-Rest (OvR)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc}")

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict()
    n_classes = len(set(y_test))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc_per_class[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for each class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label=f'ROC curve for class {i} (area = %0.2f)' % roc_auc_per_class[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Receiver Operating Characteristic (ROC) for Class {i}')
        plt.legend(loc="lower right")
        plt.show()

except AttributeError:
    print("Error: The selected model does not have a 'predict_proba' method. ROC-AUC cannot be calculated.")
except ValueError as e:
    print(f"Error calculating ROC AUC: {e}")
    print("Ensure target variable is correctly formatted for multi-class ROC-AUC or consider binary classification if applicable.")
Results
This section summarizes the performance of the applied machine learning models.

Accuracy Comparison:

Algorithm	Accuracy
Logistic Regression	[Insert Accuracy]
Decision Tree	[Insert Accuracy]
KNN	[Insert Accuracy]

Export to Sheets
(Fill in the actual accuracy values after running the code)

Individual Model Metrics (e.g., for Logistic Regression):

Precision: 0.9302
Recall: 0.9300
F1 Score: 0.9293
(These values are examples from your prompt and should be replaced with actual results from your code run.)

The graphical outputs from the code (Comparison of Algorithm Accuracies, Comparison of Algorithm Recalls, Logistic Regression Performance Metrics, and ROC for each class) provide a visual summary of the model performance.

Contributing
Feel free to contribute to this project by opening issues or submitting pull requests.
```
