import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("german_credit_data.csv")

df = df.drop('Unnamed: 0', axis=1)

df[['Saving accounts', 'Checking account']] = df[['Saving accounts', 'Checking account']].fillna('unknown')

df['Risk'] = df['Credit amount'].apply(lambda x: 'bad' if x > 3000 else 'good')

# Encode categorical variables
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode target variable: good=0, bad=1
df['Risk'] = df['Risk'].map({'good':0, 'bad':1})

# Scale numeric columns
numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Optional: check scaled values
print("\nDataset after scaling numeric features:")
print(df[numeric_cols].head())

# Features (all columns except 'Risk')
X = df.drop('Risk', axis=1)

# Target
y = df['Risk']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: check the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Get feature names
feature_names = X_train.columns

# Get model coefficients
coefficients = model.coef_[0]

# Create a DataFrame for easy plotting
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort by absolute value
coef_df['abs_coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10,7))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10))
plt.title('Top 10 Feature Coefficients (Importance)')
plt.tight_layout()
plt.show()