# 🕵️‍♂️ Fraud Detection - End-to-End ML Project (Deployed on Microsoft Azure App Service)

This project is an **end-to-end machine learning pipeline** built to predict whether a transaction is fraudulent or not based on various transaction attributes. It includes **data preprocessing, feature engineering, model training with ensembling**, and **deployment on Microsoft Azure App Service** using a Flask web interface.

---

## 🚀 Demo

Live deployment: _[Hosted on AWS Elastic Beanstalk](https://endtoendml-fraudprediction-frejcth7gugda5ay.canadacentral-01.azurewebsites.net)_  

---

## 📂 Project Structure

```
.
├── application.py              # Flask app entrypoint for AWS Elastic Beanstalk
├── artifact/                   # Contains trained model, preprocessor, and data
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── *.csv
├── catboost_info/             # Training logs for CatBoost
├── notebook/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── src/                       # Source code for ML pipeline
│   ├── components/            # Data ingestion, transformation, training modules
│   ├── pipeline/              # Prediction pipeline
│   └── utils.py, logger.py    # Utility functions and logging
├── templates/                 # HTML templates for Flask frontend
│   ├── home.html
│   └── index.html
├── requirements.txt           # Python dependencies
├── setup.py
└── README.md
```

---

## 📊 Input Features

The model expects the following input features:

| Feature                         | Description                                       |
|---------------------------------|-------------------------------------------------|
| `distance_from_home`            | Distance between user’s home and transaction      |
| `distance_from_last_transaction`| Distance from the previous transaction            |
| `ratio_to_median_purchase_price`| Ratio to median purchase price                    |
| `repeat_retailer`               | 0 or 1 (was retailer previously used)             |
| `used_chip`                     | 0 or 1 (chip used in transaction)                 |
| `used_pin_number`               | 0 or 1 (PIN was used)                             |
| `online_order`                  | 0 or 1 (whether order was online)                 |

📥 Sample input:

```csv
distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order
12,5,1.2,1,0,1,0
```

📤 Output:

```
Fraudulent / Legitimate Transaction
```

---

## 🔬 Feature Engineering

- **Transaction Distance Category**: Near / Moderate / Far
- **Purchase Price Ratio Category**: Low / Medium / High

---

## 🧠 Model Training

- Applied **SMOTE** to balance the imbalanced fraud detection dataset.
- Evaluated multiple classifiers:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Logistic Regression
  - XGBoost
  - CatBoost
  - AdaBoost
  - Voting Classifier
  - Stacking Classifier
- Best model selected based on **F1 Score**, with additional metrics:
  - ROC AUC
  - Precision
  - Recall

---

## ⚙️ Deployment

- Deployed on **Microsoft Azure App Service**

---

## 🛠️ Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/fraud-detection-ml.git
cd fraud-detection-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python application.py
```

---

## 🧪 Run Jupyter Notebook (EDA)

```bash
cd notebook
jupyter notebook
```

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).
