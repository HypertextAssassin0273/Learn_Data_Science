import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess the dataset (assuming same preprocessing as original)
data = pd.read_csv('../winequality-red.csv') # dataset in previous/parent directory

# Data Cleaning: Cap outliers using IQR method (exclude 'quality')
feature_cols = [col for col in data.columns if col != 'quality']
Q1 = data[feature_cols].quantile(0.25)
Q3 = data[feature_cols].quantile(0.75)
IQR = Q3 - Q1
for column in feature_cols:
    lower_bound = Q1[column] - 1.5 * IQR[column]
    upper_bound = Q3[column] + 1.5 * IQR[column]
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

# Data Transformation: Convert quality to categorical and scale features
data['quality_cat'] = pd.cut(data['quality'], bins=[2, 5, 8], labels=['Low', 'High'])
data = data.drop('quality', axis=1)
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Select features and target
X = data.drop('quality_cat', axis=1)
y = data['quality_cat']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels for XGBoost
y_train_encoded = y_train.map({'Low': 0, 'High': 1})
y_test_encoded = y_test.map({'Low': 0, 'High': 1})

# Step 1: Hyperparameter Tuning for Random Forest with GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, 
                              cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
rf_pred_tuned = best_rf_model.predict(X_test)

# Print results for tuned Random Forest
print("\nTuned Random Forest Best Parameters:", rf_grid_search.best_params_)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, rf_pred_tuned))
print("\nTuned Random Forest Classification Report:")
print(classification_report(y_test, rf_pred_tuned))

# Step 2: Cross-Validation for Tuned Random Forest
rf_cv_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='accuracy')
print("\nTuned Random Forest Cross-Validation Scores:", rf_cv_scores)
print("Mean CV Accuracy:", rf_cv_scores.mean())
print("Standard Deviation:", rf_cv_scores.std())

# Step 3: Ensemble Method (Voting Classifier with Random Forest, SVM, XGBoost)
svm_model = SVC(kernel='rbf', random_state=42, probability=True)  # probability=True for soft voting
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('svm', svm_model),
        ('xgb', xgb_model)
    ],
    voting='soft'  # Soft voting uses predicted probabilities
)
voting_classifier.fit(X_train, y_train_encoded)
voting_pred = voting_classifier.predict(X_test)

# Print results for Voting Classifier
print("\nVoting Classifier Accuracy:", accuracy_score(y_test_encoded, voting_pred))
print("\nVoting Classifier Classification Report:")
print(classification_report(y_test_encoded, voting_pred, target_names=['Low', 'High']))

# Step 4: Visualize Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Tuned Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred_tuned)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Tuned Random Forest Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
# Voting Classifier Confusion Matrix
cm_voting = confusion_matrix(y_test_encoded, voting_pred)
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Voting Classifier Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices_improved.png')

# Save results to a text file
with open('model_improvements_results.txt', 'w') as f:
    f.write("Tuned Random Forest Best Parameters:\n")
    f.write(str(rf_grid_search.best_params_) + "\n")
    f.write(f"Tuned Random Forest Accuracy: {accuracy_score(y_test, rf_pred_tuned)}\n")
    f.write("\nTuned Random Forest Classification Report:\n")
    f.write(classification_report(y_test, rf_pred_tuned) + "\n")
    f.write("\nTuned Random Forest Cross-Validation Scores:\n")
    f.write(str(rf_cv_scores) + "\n")
    f.write(f"Mean CV Accuracy: {rf_cv_scores.mean()}\n")
    f.write(f"Standard Deviation: {rf_cv_scores.std()}\n")
    f.write(f"\nVoting Classifier Accuracy: {accuracy_score(y_test_encoded, voting_pred)}\n")
    f.write("\nVoting Classifier Classification Report:\n")
    f.write(classification_report(y_test_encoded, voting_pred, target_names=['Low', 'High']))
