import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    if np.issubdtype(y.dtype, np.integer):
        print("Labels are already numerically encoded. No action needed.")
    else:
        print("Labels are not numerical. Applying LabelEncoder.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("\nLabelEncoder learned the following classes:")
        print(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    log_reg=LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled,y_train)

    svm_model=SVC(kernel="rbf",C=1,random_state=42,probability=True)
    svm_model.fit(X_train_scaled,y_train)

    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
    n_estimators=500,          # More trees = better performance
    max_depth=30,              # Allow deeper trees for complex relationships
    min_samples_split=5,       # Prevent overfitting
    min_samples_leaf=2,        # Each leaf must have ≥2 samples
    max_features='sqrt',       # Use sqrt of features per split (recommended)
    bootstrap=True,            # Standard bagging
    random_state=42,
    n_jobs=-1
    )

    rf_model.fit(X_train_scaled, y_train)


    y_pred_logreg=log_reg.predict(X_test_scaled)
    acc_logreg=accuracy_score(y_test,y_pred_logreg)
    print(f"Logistic Regression Accuracy: {acc_logreg * 100:.2f}%")

    y_pred_svm=svm_model.predict(X_test_scaled)
    acc_svm=accuracy_score(y_test,y_pred_svm)
    print(f"SVM Accuracy: {acc_svm * 100:.2f}%")

    acc_rf=rf_model.score(X_test_scaled,y_test)
    print(f"Random Forest Accuracy: {acc_rf * 100:.2f}%")

    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')
    print("Scaler and models have been successfully saved to disk.")
   
except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")
