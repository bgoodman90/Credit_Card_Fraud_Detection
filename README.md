# Credit Card Fraud Detection

An investigation of the **Credit Card Fraud Detection dataset** from Kaggle.  

Original dataset and variable summary:  
👉 [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Goal
- Explore and analyze the dataset  
- Identify potential issues or limitations  
- Build and evaluate machine learning models to predict fraudulent financial transactions  

---

## Repository Structure

- **`fraud_detection_initial_screen.ipynb`**  
  Main notebook containing exploratory analysis and modeling.

- **`breakdown_plots/`**  
  Contains plots generated during analysis, primarily histograms.  
  - **`Histograms/`** – Histograms of each variable in the dataset (similar to Kaggle, but recreated for hands-on analysis).  
  - **`Histograms_Downsample_Split/`** – Histograms after downsampling, split by class:  
    - `0` → Non-fraudulent transactions  
    - `1` → Fraudulent transactions  

---

## Initial Observations

- **Scope:** Data covers only 2 days and does not include individual client separation → limits long-term behavioral insights.  
- **Preprocessing:** Most features are PCA components derived from original transaction variables (e.g., type, origin country, destination country, business/personal account).  
- **Imbalance:**  
  - Non-fraudulent: **284,315**  
  - Fraudulent: **492**  
  - Fraudulent transactions make up only **0.17%** of the dataset.  
- **Quality:** No missing values. Dataset has **31 features** and **284,807 rows**.  

If original categorical variables were available, I would:  
1. One-hot encode categorical features.  
2. Perform data quality checks and feature engineering (e.g., international vs. domestic transactions).  
3. Apply PCA to reduce dimensionality and highlight differentiating features.  

---

## Handling Class Imbalance

- **Chosen method:** Downsampling → balanced dataset with 492 fraudulent + 492 non-fraudulent transactions (984 total).  
- **Rationale:**  
  - Training size is sufficient (~885 samples after 90/10 train-test split).  
  - Other options include upsampling (synthetic minority generation) or applying cost-sensitive metrics (e.g., F1 score, class weighting).  

---

## Feature Exploration & Selection

- Plotted downsampled histograms split by fraud class.  
  - Example of a **differentiating feature**:  
    ![V4 Histogram Split](https://github.com/bgoodman90/Credit_Card_Fraud_Detection/blob/main/breakdown_plots/Histograms_Downsample_Split/V4_hist.png)  
  - Example of a **non-differentiating feature**:  
    ![V22 Histogram Split](https://github.com/bgoodman90/Credit_Card_Fraud_Detection/blob/main/breakdown_plots/Histograms_Downsample_Split/V22_hist.png)  

- Used a **Random Forest classifier** to obtain feature importances:  
  ![Feature Importance](https://github.com/bgoodman90/Credit_Card_Fraud_Detection/blob/main/feature_scores.png)  

- Selected top **17 features** (cutoff at feature `V19`) for modeling.  

---

## Modeling & Results

Models were tuned via randomized CV grid search (stratified 5-fold, balanced classes, fixed random seed).  

**Cross-Validation F1-Score (avg. over 5 folds on balanced data set):**
- Random Forest → **0.9378**  
- XGBoost → **0.9425**  
- Logistic Regression → **0.9426**  
- Bernoulli Naive Bayes → **0.9036**  
- Support Vector Classifier (SVC) → **0.9393**  
- Majority Voting (XGBoost + Logistic Regression + SVC) → **0.9427 +/- 0.0098**
- Still experimenting with Neural Network, currently close to 0.88 F1-Score on test set. 

**Cross-Validation Accuracy (avg. over 5 folds on balanced data set):**
- Random Forest → **93.93%**  
- XGBoost → **94.35%**  
- Logistic Regression → **94.46%**  
- Bernoulli Naive Bayes → **91.18%**  
- Support Vector Classifier (SVC) → **94.12%**  
- Majority Voting (XGBoost + Logistic Regression + SVC) → **94.46% ± 0.97%**

I performed a quick calculation on the Logistic Regression (best performing model) to determine that AUPRC on the test set is 0.9587.

**Key Takeaways:**
- Models perform consistently around **0.94-0.95 F1-Score** and **94–95% accuracy** on the balanced dataset.  
- Results are strong given the dataset limitations and imbalance.
- Logistic Regression performs the best (Majority Voting ties it).
- Future improvements could include:  
  - Combining downsampling + upsampling  
  - Evaluating performance on full (imbalanced) data  
  - Analyzing confusion matrices (false positives vs. false negatives are critical in finance)  

---

## Summary

This quick exploration achieved ~95% accuracy using standard ML models on a balanced dataset. While strong, real-world deployment would require deeper investigation into feature engineering, class imbalance handling, and error trade-offs (false positives vs. false negatives).  
