# CreditCardFraudDetector
💳 End-to-end machine learning project for credit card fraud detection using Python, scikit-learn, and Google Colab. Includes training pipeline, evaluation reports, and an interactive demo app.

# 💳 Credit Card Fraud Detection

Fraudulent credit card transactions represent **less than 0.2% of all activity**, yet the cost of missing them can be enormous.  
This project implements a **machine learning pipeline** that detects fraud with high recall, using the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The project was developed in **Google Colab** and includes training, evaluation, and an interactive app for testing transactions.

---

## 🚀 Key Features
- Handles **imbalanced data** with class weighting  
- Compares **Logistic Regression** and **Random Forest** classifiers  
- Evaluates with **Precision–Recall AUC** (more informative than accuracy for rare events)  
- Tunes the decision threshold for **high fraud recall (~90%)**  
- Saves trained pipeline + threshold as artifacts  
- Generates performance reports:
  - Precision–Recall curve
  - ROC curve
  - Confusion Matrix  
- Includes an interactive **Gradio app** to upload and score CSVs  

---

## 📊 Dataset
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Contains anonymized credit card transactions with 30 features (`V1`–`V28`, `Time`, `Amount`) and labels (`Class`: 0 = legit, 1 = fraud).  
- ⚠️ Dataset is **not included** in this repository due to size. Please download it manually and place it in `data/creditcard.csv`.

---

## 📂 Project Structure
fraud-detector/
├─ data/ # dataset (not included in repo)
│ └─ README.md
├─ models/ # trained pipeline + threshold (created after training)
│ └─ README.md
├─ reports/ # output plots & metrics (created after training)
│ └─ README.md
├─ notebooks/
│ └─ fraud_detector_colab.ipynb # Colab notebook version
├─ train.py # training script
├─ app_gradio.py # demo app for scoring CSVs
├─ requirements.txt # dependencies
└─ README.md

yaml
Copy code

---

## ▶️ Quickstart

### 1. Clone repository
```bash
git clone https://github.com/YOURUSERNAME/fraud-detector.git
cd fraud-detector
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Download dataset
Get the dataset from Kaggle: Credit Card Fraud Dataset

Place the CSV file at: data/creditcard.csv

4. Train the model
bash
Copy code
python train.py --data data/creditcard.csv --out .
Artifacts will be saved in models/ and evaluation plots in reports/.

5. Run the demo app
bash
Copy code
python app_gradio.py
Upload a CSV of transactions to see fraud probabilities and flagged transactions.

📷 Example Outputs
Precision–Recall Curve

Confusion Matrix

Gradio Demo

🔧 Tools & Libraries
Python

scikit-learn

pandas, numpy

matplotlib

Gradio

Google Colab

📈 Results
Best model: Random Forest (selected by validation PR–AUC)

Threshold tuning: Optimized for high fraud recall (~90%)

Average Precision (AP): XX% (on test set)

📌 Future Improvements
Experiment with SMOTE or other resampling methods

Add XGBoost / LightGBM classifiers

Deploy demo via Streamlit Cloud

Implement cost-sensitive evaluation (different penalties for FP vs FN)

