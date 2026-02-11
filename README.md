🏥 Hospital Length of Stay Prediction

This project focuses on predicting a patient’s length of hospital stay using structured healthcare data.
It covers end-to-end machine learning workflow including data cleaning, preprocessing, feature engineering, model training, and evaluation.

📊 Dataset Overview

Total records: 318,438

Unique patients: 318,437

Target variable: Stay (11 classes representing length of stay ranges).

Key Features

Hospital details (code, region, type)

Patient demographics (age, city code)

Admission details (type, severity, visitors)

Ward and department information

Admission deposit

🧹 Data Cleaning & Preprocessing
Missing Value Handling

Identified missing values using:

data.isnull().sum()


Columns with missing values:

Bed Grade: 113 missing values

City_Code_Patient: 4532 missing values

Filled missing values using mean imputation.

def preprocessing(data, columns):
    for column in columns:
        data[column] = data[column].fillna(data[column].mean())

🔁 Categorical Feature Encoding
One-Hot Encoding (Unordered Categories)

Applied to:

Hospital_type_code

Hospital_region_code

Department

Ward_Type

Ward_Facility_Code

pd.get_dummies()

Label Encoding (Ordered Categories)

Applied to:

Type of Admission

Severity of Illness

Age

Stay (Target)

Custom ordinal mappings were defined to preserve meaningful order.

Example (Stay mapping):

{'0-10': 0, '11-20': 1, ..., 'More than 100 Days': 10}

📐 Feature Scaling

Used StandardScaler to normalize features

Ensures unit variance and improves model performance

scaler = StandardScaler()
X = scaler.fit_transform(X)

🧠 Machine Learning Models Used

The following classification models were trained and evaluated:

Logistic Regression

Multilayer Perceptron (MLP)

Random Forest Classifier

Gaussian Naive Bayes

K-Nearest Neighbors

Decision Tree Classifier

Train-test split:

train_size = 80%
test_size = 20%

📈 Model Performance Comparison
Model	Accuracy
Logistic Regression	38.70%
MLP Classifier	41.92% (Best)
Random Forest	39.75%
KNN	32.84%
Decision Tree	30.13%
Gaussian Naive Bayes	10.59%
📋 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix (saved as images)

Example:

classification_report(y_test, y_pred)
plot_confusion_matrix(model, X_test, y_test)

🔍 Key Observations

The dataset is highly imbalanced, affecting minority class predictions

MLP performed best among all models

Gaussian Naive Bayes struggled due to feature assumptions

Ordinal encoding significantly improved interpretability

🚀 Future Improvements

Handle class imbalance using SMOTE or class weighting

Hyperparameter tuning (GridSearch / RandomSearch)

Try Gradient Boosting models (XGBoost, LightGBM)

Feature selection & dimensionality reduction

Convert to regression or ordinal classification approach

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Jupyter Notebook
