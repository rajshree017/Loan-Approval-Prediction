"""
╔══════════════════════════════════════════════════════╗
║         LOAN APPROVAL PREDICTION                     ║
║   Python | Pandas | Sklearn | Matplotlib | Seaborn   ║
╚══════════════════════════════════════════════════════╝

Predicts whether a loan will be approved or rejected
based on applicant's financial and personal details.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
          '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# ============================================================
#  GENERATE SAMPLE DATA
# ============================================================
def generate_data():
    np.random.seed(42)
    n = 1000

    gender         = np.random.choice(['Male', 'Female'], n, p=[0.65, 0.35])
    married        = np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35])
    dependents     = np.random.choice(['0', '1', '2', '3+'], n, p=[0.57, 0.17, 0.16, 0.10])
    education      = np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22])
    self_employed  = np.random.choice(['Yes', 'No'], n, p=[0.14, 0.86])
    applicant_income   = np.random.randint(1000, 20000, n)
    coapplicant_income = np.random.randint(0, 10000, n)
    loan_amount    = np.random.randint(50, 700, n)
    loan_term      = np.random.choice([120, 180, 240, 300, 360, 480], n)
    credit_history = np.random.choice([0, 1], n, p=[0.16, 0.84])
    property_area  = np.random.choice(['Urban', 'Semiurban', 'Rural'], n)

    # Loan approval logic
    score = (credit_history * 40 +
             (applicant_income > 5000).astype(int) * 20 +
             (loan_amount < 300).astype(int) * 15 +
             (education == 'Graduate').astype(int) * 10 +
             (married == 'Yes').astype(int) * 10 +
             np.random.randint(0, 20, n))

    loan_status = (score > 60).astype(int)

    df = pd.DataFrame({
        'Gender'            : gender,
        'Married'           : married,
        'Dependents'        : dependents,
        'Education'         : education,
        'Self_Employed'     : self_employed,
        'ApplicantIncome'   : applicant_income,
        'CoapplicantIncome' : coapplicant_income,
        'LoanAmount'        : loan_amount,
        'Loan_Amount_Term'  : loan_term,
        'Credit_History'    : credit_history,
        'Property_Area'     : property_area,
        'Loan_Status'       : loan_status
    })
    return df

# ============================================================
#  LOAD DATA
# ============================================================
def load_data():
    print("\n📂 Loading Loan Dataset...")
    try:
        df = pd.read_csv('loan_data.csv')
        print(f"✅ Dataset loaded from CSV: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠️  No CSV found — generating sample data...")
        df = generate_data()
        df.to_csv('loan_data.csv', index=False)
        print(f"✅ Sample dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

# ============================================================
#  DATA CLEANING & ENCODING
# ============================================================
def preprocess_data(df):
    print("\n🧹 Preprocessing Data...")
    df = df.copy()
    df.fillna(df.mode().iloc[0], inplace=True)

    le = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"✅ Data preprocessed: {df.shape}")
    return df

# ============================================================
#  ANALYSIS 1: Loan Approval Distribution
# ============================================================
def analysis_approval_distribution(df):
    print("\n📊 Analysis 1: Loan Approval Distribution...")
    counts = df['Loan_Status'].value_counts()
    labels = ['Approved ✅', 'Rejected ❌']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    axes[0].pie(counts.values, labels=labels,
                autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'],
                startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Loan Approval Rate', fontsize=14, fontweight='bold')

    # Bar chart
    bars = axes[1].bar(labels, counts.values,
                       color=['#4ECDC4', '#FF6B6B'], edgecolor='white')
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5, str(val),
                     ha='center', fontweight='bold', fontsize=12)
    axes[1].set_title('Approval Count', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')

    plt.suptitle('💰 Loan Approval Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('1_approval_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 1_approval_distribution.png")

# ============================================================
#  ANALYSIS 2: Credit History Impact
# ============================================================
def analysis_credit_history(df):
    print("\n📊 Analysis 2: Credit History Impact...")
    credit_approval = df.groupby('Credit_History')['Loan_Status'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars = axes[0].bar(['Bad Credit (0)', 'Good Credit (1)'],
                       credit_approval.values,
                       color=['#FF6B6B', '#4ECDC4'], edgecolor='white')
    for bar, val in zip(bars, credit_approval.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center',
                     fontweight='bold', fontsize=12)
    axes[0].set_title('Credit History vs Approval Rate',
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Approval Rate (%)')

    # Income distribution
    approved = df[df['Loan_Status'] == 1]['ApplicantIncome']
    rejected = df[df['Loan_Status'] == 0]['ApplicantIncome']
    axes[1].hist(approved, bins=30, alpha=0.7, color='#4ECDC4',
                 label='Approved', edgecolor='white')
    axes[1].hist(rejected, bins=30, alpha=0.7, color='#FF6B6B',
                 label='Rejected', edgecolor='white')
    axes[1].set_title('Income Distribution by Status',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Applicant Income')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.suptitle('📊 Key Factor Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('2_credit_income.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 2_credit_income.png")

# ============================================================
#  ANALYSIS 3: Categorical Features vs Approval
# ============================================================
def analysis_categorical(df_original):
    print("\n📊 Analysis 3: Categorical Features vs Approval...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cat_features = ['Gender', 'Married', 'Education', 'Property_Area']

    for i, feature in enumerate(cat_features):
        approval_rate = df_original.groupby(feature)['Loan_Status'].mean() * 100
        bars = axes[i].bar(approval_rate.index, approval_rate.values,
                           color=COLORS[:len(approval_rate)], edgecolor='white')
        for bar, val in zip(bars, approval_rate.values):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.5,
                         f'{val:.1f}%', ha='center',
                         fontweight='bold', fontsize=10)
        axes[i].set_title(f'{feature} vs Approval Rate',
                          fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Approval Rate (%)')
        axes[i].set_ylim(0, 100)

    plt.suptitle('📈 Categorical Features vs Loan Approval',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('3_categorical_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 3_categorical_analysis.png")

# ============================================================
#  ML: LOGISTIC REGRESSION
# ============================================================
def train_model(df):
    print("\n🤖 Training Logistic Regression Model...")

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n✅ Model Performance:")
    print(f"   Accuracy  : {acc:.2f}%")
    print(f"   ROC-AUC   : {auc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Rejected', 'Approved']))

    return model, scaler, X_test, y_test, y_pred, y_pred_prob, acc, auc

# ============================================================
#  ANALYSIS 4: Confusion Matrix
# ============================================================
def analysis_confusion_matrix(y_test, y_pred, acc):
    print("\n📊 Analysis 4: Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'],
                ax=ax, linewidths=1)
    ax.set_title(f'🎯 Confusion Matrix (Accuracy: {acc:.2f}%)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('4_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 4_confusion_matrix.png")

# ============================================================
#  ANALYSIS 5: ROC Curve
# ============================================================
def analysis_roc_curve(y_test, y_pred_prob, auc):
    print("\n📊 Analysis 5: ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#4ECDC4', linewidth=2.5,
            label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#FF6B6B', linewidth=1.5,
            linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#4ECDC4')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('📈 ROC Curve - Logistic Regression',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('5_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 5_roc_curve.png")

# ============================================================
#  PREDICT NEW APPLICANT
# ============================================================
def predict_applicant(model, scaler, df):
    print("\n💰 Sample Loan Prediction:")
    sample = df.drop('Loan_Status', axis=1).iloc[0:1]
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1] * 100
    print(f"   Prediction  : {'✅ APPROVED' if prediction == 1 else '❌ REJECTED'}")
    print(f"   Probability : {probability:.2f}% chance of approval")

# ============================================================
#  SUMMARY
# ============================================================
def print_summary(df, acc, auc):
    print("\n" + "="*55)
    print("       💰 LOAN APPROVAL PREDICTION SUMMARY")
    print("="*55)
    total    = len(df)
    approved = df['Loan_Status'].sum()
    rejected = total - approved
    print(f"  Total Applications : {total:,}")
    print(f"  Approved           : {approved:,} ({approved/total*100:.1f}%)")
    print(f"  Rejected           : {rejected:,} ({rejected/total*100:.1f}%)")
    print(f"  Model              : Logistic Regression")
    print(f"  Accuracy           : {acc:.2f}%")
    print(f"  ROC-AUC Score      : {auc:.4f}")
    print("="*55)

# ============================================================
#  MAIN
# ============================================================
def main():
    print("="*55)
    print("       💰 LOAN APPROVAL PREDICTION")
    print("   Python | Pandas | Sklearn | Matplotlib")
    print("="*55)

    df_original = load_data()
    df          = preprocess_data(df_original)

    # Analyses
    analysis_approval_distribution(df_original)
    analysis_credit_history(df_original)
    analysis_categorical(df_original)

    # ML
    model, scaler, X_test, y_test, y_pred, y_pred_prob, acc, auc = train_model(df)
    analysis_confusion_matrix(y_test, y_pred, acc)
    analysis_roc_curve(y_test, y_pred_prob, auc)
    predict_applicant(model, scaler, df)

    print_summary(df_original, acc, auc)
    print("\n✅ All analyses complete!")
    print("📁 5 charts saved as PNG files!")

if __name__ == "__main__":
    main()