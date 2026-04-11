import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. LOAD DATA
# Fetching the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['Ciaze', 'Glukoza', 'Cisnienie', 'Skora', 'Insulina', 'BMI', 'Rodowod', 'Wiek', 'Wynik']
df = pd.read_csv(url, names=names)

# 2. DATA CLEANING & PREPROCESSING
# Replace biologically impossible zero values with NaN, then impute with the median
for col in ['Glukoza', 'Cisnienie', 'Skora', 'Insulina', 'BMI']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Define a function to remove extreme outliers using the Interquartile Range (IQR) method
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    # Filter rows within the acceptable range
    return data[(data[column] >= Q1 - 1.5*IQR) & (data[column] <= Q3 + 1.5*IQR)]

# Apply outlier removal to critical features
df = remove_outliers(df, 'BMI')
df = remove_outliers(df, 'Glukoza')

# 3. DATA SPLITTING
# Separate features (X) and target variable (y)
X = df.drop('Wynik', axis=1)
y = df['Wynik']

# Split into training and testing sets (80/20). 'stratify=y' maintains class distribution.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. MODEL TRAINING
# Initialize Decision Tree with max_depth=4 to prevent overfitting
# 'class_weight=balanced' helps the model prioritize the minority class (diabetic patients)
tree_model = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
tree_model.fit(X_train, y_train)

# 5. CUSTOM THRESHOLD CLASSIFICATION
# Lower the decision threshold to prioritize medical sensitivity (Recall)
THRESHOLD = 0.4
y_probs = tree_model.predict_proba(X_test)[:, 1] # Get probabilities for class 1
y_pred = (y_probs > THRESHOLD).astype(int)       # Apply custom threshold

# 6. EVALUATION & REPORT
print("--- SYSTEM DIAGNOSTYCZNY AI: WYNIKI ---")
print(f"Ogólna dokładność: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"\nSzczegółowy raport klasyfikacji (Próg = {THRESHOLD}):")
print(classification_report(y_test, y_pred))

# 7. INDIVIDUAL VISUALIZATIONS

# --- Plot 1: Decision Tree Structure ---
# Visualizing the AI logic path
plt.figure(figsize=(22, 12))
plot_tree(tree_model,
          feature_names=X.columns,
          class_names=['Zdrowy', 'Chory'],
          filled=True,
          rounded=True,
          fontsize=11)
plt.title("Mapa myśli AI (Drzewo Decyzyjne)", fontsize=16)
plt.tight_layout()
plt.show()

# --- Plot 2: Confusion Matrix ---
# Visualizing True/False Positives and Negatives
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges', annot_kws={"size": 14})
plt.title(f'Macierz Pomyłek (Próg {THRESHOLD})', fontsize=14)
plt.ylabel('Stan faktyczny', fontsize=12)
plt.xlabel('Diagnoza AI', fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot 3: Feature Importance ---
# Displaying which parameters had the highest impact on the model's decisions
plt.figure(figsize=(10, 6))
importances = pd.Series(tree_model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='darkorange')
plt.title('Które parametry najbardziej wpływają na diagnozę?', fontsize=14)
plt.xlabel('Znaczenie cechy', fontsize=12)
plt.ylabel('Parametry medyczne', fontsize=12)
plt.tight_layout()
plt.show()