# Customer Churn Prediction Machine Learning Model

### Description
Customer retention is a critical metric for business growth and sustainability. This project aims to address the challenge of customer attrition by building a predictive machine learning pipeline that identifies customers who are likely to cancel or leave a service ("Churn"). 

The implementation features an end-to-end data pipeline, including target label encoding (`Yes` to `1`, `No` to `0`), handling class imbalance using **SMOTE**, and evaluating multiple tree-based ensemble models. After evaluating performance using stratified cross-validation, the optimal model is serialized for future production use.

---

### Key Features
* **Target Label Optimization:** Automates the mapping of text labels (`Yes`/`No`) into a binary machine-readable format.
* **Class Imbalance Handling:** Employs **SMOTE** (`X_train_smote`, `Y_train_smote`) to oversample the minority churn class, ensuring the models are not biased toward non-churning customers.
* **Multi-Model Evaluation:** Trains and evaluates three robust classifiers:
  * Decision Tree
  * Random Forest (Top Performer: ~84% Mean Cross-Validation Accuracy)
  * XGBoost
* **Comprehensive Metrics Logging:** Features a thorough evaluation phase assessing Accuracy, Confusion Matrices, and Classification Reports (Precision, Recall, F1-Score).
* **Model Serialization:** Exports the final trained model alongside its verified feature names into a reusable pickle file (`customer_churn_model.pkl`).

---

### Technologies Used
* **Language:** Python 3
* **Data Manipulation:** NumPy, Pandas
* **Machine Learning Framework:** Scikit-Learn (`tree`, `ensemble`, `model_selection`, `metrics`)
* **Gradient Boosting:** XGBoost (`XGBClassifier`)
* **Imbalanced Data Handling:** Imbalanced-Learn (`SMOTE`)
* **Model Artifact Deployment:** Pickle

---

### How to Run

#### 1. Prerequisites
Ensure you have Python 3 installed. Install the required dependencies using pip:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn
