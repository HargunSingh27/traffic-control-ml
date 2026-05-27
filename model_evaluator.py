# Import necessary libraries
import pandas as pd
import numpy as np
import joblib 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Model Evaluation Script ---")

try:
    print("\n[INFO] Assuming data, models, and predictions are loaded...")
    features_df = pd.read_csv("features.csv")
    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = joblib.load('scaler.joblib')
    log_reg_model = joblib.load('logistic_regression_model.joblib')
    svm_model = joblib.load('svm_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')
    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5', compile=False)

    X_test_scaled = scaler.transform(X_test)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

    genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                   'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)
    print("\nConfusion matrices computed successfully.")

    def plot_confusion_matrix(cm, labels, title, ax):
        sns.heatmap(
            cm,
            annot=True,          
            fmt='d',             
            cmap='Blues',        
            xticklabels=labels,  
            yticklabels=labels,  
            ax=ax                
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for All Models', fontsize=20)

    plot_confusion_matrix(cm_log_reg, genre_names, 'Logistic Regression', axes[0, 0])
    plot_confusion_matrix(cm_svm, genre_names, 'Support Vector Machine', axes[0, 1])
    plot_confusion_matrix(cm_rf, genre_names, 'Random Forest', axes[1, 0])
    plot_confusion_matrix(cm_cnn, genre_names, 'Convolutional Neural Network', axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

except FileNotFoundError as e:
    print(f"\nERROR: A required file was not found: {e.filename}")
    print("Please ensure all model files ('scaler.joblib', '*.joblib', 'music_genre_cnn.h5') and 'features.csv' are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
