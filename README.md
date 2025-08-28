# 📊 Customer Churn Prediction

This project predicts whether a customer will **churn** (leave the company) or **stay**, based on demographic, service usage, and account information.  
It uses **synthetic data generation**, **feature encoding**, and **machine learning models** like **Logistic Regression**.

---

## 📂 Project Structure
```

├── churn\_prediction.ipynb   # Jupyter notebook with full workflow
├── churn\_app.py              # (Optional) Streamlit app for interactive prediction
├── churn\_data.csv            # Generated dataset (500 rows)
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation

````

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/churn-prediction.git
   cd churn-prediction
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📑 Requirements

Key dependencies:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
ipywidgets
streamlit   # (only if running app)
```

---

## 🚀 Usage

### 1. Generate Dataset

In Jupyter Notebook:

```python
from data_generator import generate_data
df = generate_data(500)
df.to_csv("churn_data.csv", index=False)
```

### 2. Train Model

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

# Encode + train
...
```

### 3. Evaluate Model

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:", classification_report(y_test, y_pred))
```

### 4. Visualize

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Churn", data=df)
plt.show()
```

### 5. Run Streamlit App (optional)

```bash
streamlit run churn_app.py
```

---

## 📊 Example Output

* **Accuracy:** \~80-85% (varies with random generation)
* **Confusion Matrix** and **Classification Report**
* Interactive plots in notebook & Streamlit dashboard

---

## ✨ Features

* Generates **realistic telecom customer data**
* Preprocessing with **Label Encoding**
* Train/Test split and **Logistic Regression model**
* Visualizations with **Matplotlib & Seaborn**
* Interactive **Streamlit App** for live predictions

---

## 📌 Next Steps

* Add **XGBoost / RandomForest** for better accuracy
* Save & load trained model with `pickle`
* Deploy app to **Streamlit Cloud / Heroku**

---

## 👨‍💻 Author

Developed by **\[KISHORE HUI]** ✨
