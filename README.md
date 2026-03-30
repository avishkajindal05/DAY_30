# DAY_30
DAY-30-AM
# 🚗 SUV Purchase Prediction using Logistic Regression

## 📌 Project Overview

This project implements **Logistic Regression** using `sklearn` to predict whether a customer will purchase an SUV based on demographic features.

The dataset used is a typical **SUV dataset**, where the model learns patterns between user attributes and purchase decisions.

---

## 📂 Repository Structure

```
DAY_30/
│
├── suv_data.csv                         # Dataset
├── suv_logistic_regression_partA.ipynb  # Data preprocessing + training
├── suv_logistic_regression_partB.ipynb  # Evaluation + insights
├── PART-c.md                           # Theory / explanations
├── PART-d.ipynb                        # Text-based answers (no code)
├── README.md                           # Project documentation
├── LICENSE
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/avishkajindal05/DAY_30.git
cd DAY_30
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook
```

---

## ▶️ How to Run

### Step 1: Open Notebook

Run the following notebooks in order:

1. `suv_logistic_regression_partA.ipynb`

   * Data loading
   * Train-test split
   * Feature scaling
   * Model training

2. `suv_logistic_regression_partB.ipynb`

   * Predictions
   * Confusion matrix
   * Accuracy evaluation

3. `PART-d.ipynb`

   * Contains **theoretical answers (text only)** for submission

---

## 🧠 Model Details

* Algorithm: **Logistic Regression**
* Type: **Supervised Learning (Classification)**
* Target Variable: `Purchased` (0 or 1)

### Key Steps:

* Data preprocessing
* Feature scaling using `StandardScaler`
* Model training using `LogisticRegression`
* Performance evaluation using:

  * Confusion Matrix
  * Accuracy Score

---

## 📊 Evaluation Metrics

* **Confusion Matrix**
* **Accuracy**
* (Optional extension: Precision, Recall, F1-score)

---

## 📈 Insights

* Age and Salary significantly influence SUV purchase decisions
* Logistic Regression provides a **linear decision boundary**
* Model outputs **probabilities**, not direct classes

---

## 📌 Notes

* Scaling is crucial for Logistic Regression performance
* Training and testing data must be handled separately to avoid leakage
* `PART-d.ipynb` is intentionally text-based for theoretical answers

---

## 🏁 Conclusion

This project demonstrates a complete ML pipeline:

* Data preprocessing
* Model training
* Evaluation
* Interpretation

It is a foundational example of applying **classification algorithms in real-world scenarios**.

---

## 📜 License

This project is licensed under the MIT License.
