import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Configuration
CSV_PATH = "customer_booking.csv" 
TARGET_COL = 'booking_complete'
OPTIMAL_THRESHOLD = 0.20

# Load dataset
df = pd.read_csv(CSV_PATH, encoding='gbk')
print(f"Dataset shape: {df.shape}\n")

# Separate features and target
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

# Stratified train-test split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing and modeling pipeline
numeric_tf = SimpleImputer(strategy='median')
categorical_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_tf, num_cols),
    ('cat', categorical_tf, cat_cols)
])

# Initialize Random Forest with balanced class weights
model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring_metrics, n_jobs=-1)

print("=== Cross-Validation Results (Mean) ===")
for metric in scoring_metrics:
    print(f"{metric.capitalize()}: {cv_res[f'test_{metric}'].mean():.4f}")

# Train the final model and predict probabilities
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# Threshold tuning experiment
print("\n=== Threshold Optimization Experiment ===")
thresholds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
for thr in thresholds:
    y_pred_thr = (y_prob >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_thr, average='binary')
    print(f"Threshold={thr:.2f} -> Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

# Evaluate model at the optimal threshold
y_pred_optimal = (y_prob >= OPTIMAL_THRESHOLD).astype(int)
print(f"\n=== Final Evaluation on Test Set (Threshold = {OPTIMAL_THRESHOLD}) ===")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_optimal, average='binary')
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# Generate and save Confusion Matrix (using optimal threshold)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Threshold = {OPTIMAL_THRESHOLD})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()

# Generate and save Feature Importance plot
rf_model = model.named_steps['rf']
ohe = model.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
ohe_columns = ohe.get_feature_names_out(cat_cols) if len(cat_cols) > 0 else []
feature_names = np.concatenate([num_cols, ohe_columns])

importances = pd.Series(rf_model.feature_importances_, index=feature_names)
top_feats = importances.sort_values(ascending=False).head(20)

top_feats.plot(kind='barh', figsize=(10, 6))
plt.title("Top 20 Feature Importances (Random Forest)")
plt.gca().invert_yaxis() # Display the most important feature at the top
plt.tight_layout()
plt.savefig("feature_importance.png")

print("\nPlots have been successfully saved as 'confusion_matrix.png' and 'feature_importance.png'.")