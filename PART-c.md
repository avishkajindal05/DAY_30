## **Q1 — What is Logistic Regression? Is it classification or regression?**

**Logistic Regression** is a supervised machine learning algorithm used to model the probability of a **categorical dependent variable** (usually binary: 0 or 1).

* It uses a **logistic (sigmoid) function** to map predicted values into a probability between 0 and 1.
* The output is interpreted as:
  [
  P(y=1 \mid x)
  ]

**Key idea:**
Instead of predicting a continuous value (like linear regression), it predicts **probability**, which is then converted into a class label using a threshold (e.g., 0.5).

**Why “Regression” in name?**

* It models the **log-odds (logit)** as a linear combination of inputs:
  [
  \log\left(\frac{p}{1-p}\right) = w^Tx + b
  ]

### ✅ Final Answer:

* Logistic Regression is technically a **regression model** (because it models log-odds),
* But it is **used for classification tasks**.

---

## **Q2 — Code: Train-Test Split and Scaling**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume X = features, y = target

# Step 1: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 2: Feature Scaling
scaler = StandardScaler()

# Fit on training data and transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Key Points:

* **train_test_split** prevents data leakage.
* **fit only on training data**, transform both train & test.
* Scaling is important for distance-based and gradient-based models.

---

## **Q3 — What is a Confusion Matrix? What does it represent?**

A **confusion matrix** is a table used to evaluate the performance of a **classification model** by comparing **actual vs predicted values**.

### Structure (Binary Classification):

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

### What it represents:

* **TP** → Correctly predicted positives
* **TN** → Correctly predicted negatives
* **FP** → Incorrectly predicted positives (Type I error)
* **FN** → Incorrectly predicted negatives (Type II error)

### Why it’s important:

From this matrix, we derive key metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**

### Insight:

It gives a **complete picture of model performance**, especially when dealing with **imbalanced datasets**, where accuracy alone can be misleading.

