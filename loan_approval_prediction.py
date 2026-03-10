# Loan Approval Prediction using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Generate sample dataset since we don't have a CSV
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'Gender'            : np.random.choice(['Male', 'Female'], n),
    'Married'           : np.random.choice(['Yes', 'No'], n),
    'Education'         : np.random.choice(['Graduate', 'Not Graduate'], n),
    'ApplicantIncome'   : np.random.randint(1000, 20000, n),
    'LoanAmount'        : np.random.randint(50, 700, n),
    'Credit_History'    : np.random.choice([0, 1], n, p=[0.16, 0.84]),
    'Property_Area'     : np.random.choice(['Urban', 'Semiurban', 'Rural'], n),
})

# Create loan status based on some logic
score = (df['Credit_History'] * 40 +
         (df['ApplicantIncome'] > 5000).astype(int) * 20 +
         (df['LoanAmount'] < 300).astype(int) * 15 +
         (df['Education'] == 'Graduate').astype(int) * 10 +
         np.random.randint(0, 20, n))

df['Loan_Status'] = (score > 60).astype(int)

print("Dataset Shape:", df.shape)
print(df.head())

# --- Chart 1: Loan Approval Count ---
counts = df['Loan_Status'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(['Rejected', 'Approved'], counts.values, color=['salmon', 'steelblue'], edgecolor='white')
plt.title('Loan Approval vs Rejection Count')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('1_approval_count.png')
plt.show()

# --- Chart 2: Credit History vs Approval Rate ---
credit_approval = df.groupby('Credit_History')['Loan_Status'].mean() * 100
plt.figure(figsize=(6, 4))
plt.bar(['Bad Credit', 'Good Credit'], credit_approval.values, color=['salmon', 'steelblue'])
plt.title('Credit History vs Loan Approval Rate')
plt.ylabel('Approval Rate (%)')
plt.tight_layout()
plt.savefig('2_credit_history.png')
plt.show()

# --- Chart 3: Income Distribution by Loan Status ---
plt.figure(figsize=(8, 5))
df[df['Loan_Status'] == 1]['ApplicantIncome'].hist(bins=30, alpha=0.6, color='steelblue', label='Approved')
df[df['Loan_Status'] == 0]['ApplicantIncome'].hist(bins=30, alpha=0.6, color='salmon', label='Rejected')
plt.title('Income Distribution - Approved vs Rejected')
plt.xlabel('Applicant Income')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('3_income_distribution.png')
plt.show()

# Encode categorical columns
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Property_Area']:
    df[col] = le.fit_transform(df[col])

# Prepare features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
acc = accuracy_score(y_test, y_pred) * 100
auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy  : {acc:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# --- Chart 4: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'])
plt.title(f'Confusion Matrix (Accuracy: {acc:.2f}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('4_confusion_matrix.png')
plt.show()

# --- Chart 5: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('5_roc_curve.png')
plt.show()

print("\nDone! All charts saved.")