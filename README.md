# ML-Based Adaptive Traffic Control System

A machine learning system that predicts traffic volume and adapts signal timings based on traffic density to reduce congestion and improve vehicle flow efficiency.

## Features
- Traffic volume prediction using multiple ML models (Logistic Regression, SVM, CNN)
- Exploratory Data Analysis (EDA) on real-world traffic dataset
- Feature extraction and preprocessing pipeline
- Model evaluation with accuracy and performance metrics

## Tech Stack
- Python
- Scikit-learn (Logistic Regression, SVM)
- TensorFlow / Keras (CNN)
- Pandas, NumPy
- Matplotlib / Seaborn

## Project Structure
ML_project/
├── mlprojfinal.ipynb          # Main notebook
├── app.py                     # Application entry point
├── eda.py                     # Exploratory data analysis
├── feature_extractor.py       # Feature engineering
├── model_trainer.py           # Model training scripts
├── model_evaluator.py         # Evaluation metrics
├── check_dataset.py           # Dataset validation
├── Metro_Interstate_Traffic_Volume.csv  # Dataset
└── features.csv               # Extracted features
## Dataset
Metro Interstate Traffic Volume dataset — contains hourly traffic volume data with weather and holiday features.

## How to Run

1. Clone the repo
```bash
git clone https://github.com/HargunSingh27/traffic-control-ml.git
cd traffic-control-ml
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the notebook
```bash
jupyter notebook mlprojfinal.ipynb
```

4. Or run the app directly
```bash
python app.py
```

## Results
- Multiple ML models trained and evaluated on traffic volume prediction
- Best models serialized using joblib for deployment
- CNN model trained for pattern recognition in traffic data

## Author
**Hargun Singh**
B.E. Computer Engineering, Thapar Institute of Engineering & Technology
[LinkedIn](https://linkedin.com/in/hargun-singh-4b0344372) | [GitHub](https://github.com/HargunSingh27)
