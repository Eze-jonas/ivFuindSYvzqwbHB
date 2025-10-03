### Smarter Customer Surveys: Using Machine Learning to Improve Service Quality
**Context:**
* Project completed while at Apziva. This repository contains exploration, preprocessing, and model development.
* It is Aimed at predicting customer happiness, Improve quality of customer services and optimizing survey design using a happiness datasetfrom a logistics/delivery company.
* The work highlights how machine learning can:
* **Rank survey questions by importance**, enabling companies to focus their improvements in the areas that matter most to customers.
* **Identify the most impactful survey questions that preserve accuracy and maximise prediction**, helping businesses shorten surveys, reduce customer effort, and still capture actionable insights.
* All notebooks are reproducible, auditable, and designed to generate artifacts that can be integrated into a production pipeline.
---

## Table of contents
1. Project summary
2. project Achievement
3. Highlights & dataset
6. Step-by-step project flow
7. Technical details & explanations
8. How to reproduce locally
9. Repository structure
10. What I contributed / hiring-manager summary
11. Next steps & suggestions
---

## Project summary

* This project used customer survey responses to build a classification model that predicts whether a customer is *happy or not* after receiving logistics/delivery services.  
* Identified the most important survey questions/features that drive customer happiness.   
* Determined the minimal subset of features that preserves accuracy and maximizes predictability, ensuring future surveys can be shorter, focused, and less burdensome while retaining diagnostic power. 
* The work includes exploratory data analysis (EDA), robust preprocessing (a reusable pipeline), model selection and hyperparameter tuning, and the creation of saved preprocessing artifacts for reproducibility.

## Project Achievements:

1. Built Two Random Forest Models (Not Deployment-Ready)

i. **Random Forest for Feature Importance**
* This identified the most important survey questions/features that drive customer happiness.
* **Setup**: LOOCV within GridSearchCV.
###### **Evaluation**:
###### Full dataset (optimistic): Accuracy: 0.836, Precision: 0.828, Recall: 0.857, F1 Score: 0.842, AUC: 0.926
###### LOOCV (realistic): Accuracy: 0.546, Precision: 0.554, Recall: 0.554, F1 Score: 0.554, AUC: 0.551
###### Note: Although the model overfits on the full dataset, it was primarily used to extract feature importance rather than for deployment.

##### ii. **Random Forest for Feature Selection**
* This determined the minimal subset of features (K = 3) that preserves accuracy and maximizes predictability, ensuring future surveys can be shorter, focused, and less burdensome while retaining diagnostic power.
* **Setup**: 10-Fold Stratified CV within GridSearchCV, applied SelectKBest.
###### **Evaluation**:
* Full dataset (optimistic): Accuracy: 0.773, Precision: 0.763, Recall: 0.804, F1 Score: 0.783, AUC: 0.853
* 10-Fold Stratified CV (realistic): Accuracy: 0.655, Precision: 0.655, Recall: 0.679, F1 Score: 0.667, AUC: 0.646  
* The moderate drop in metrics shows the model is not heavily overfitting, supporting the validity of feature selection and the model’s ability to capture meaningful patterns from the survey responses.
2. Produced three main reproducible artifacts:
* EDA notebook – detailed exploratory analysis of survey data.
* Preprocessing pipeline – automated EDA steps for reuse.
* Model development notebook – end-to-end workflow for training, hyperparameter tuning, evaluation, feature importance extraction, feature selection (SelectKBest), and final project summary.
---
## Highlights & dataset
* The project used a survey dataset called ACME-HappinessSurvey2020.csv(not included in this repo for privacy) from a logistics/delivery company.
* The dataset has 126 survey responses (rows) and 7 columns:
* Y – binary target (customer happy = 1, not happy = 0)
* X1 to X6 – numerical features(survey question responses)
* After removing duplicated rows, the dataset was reduced from 126 to 110
* Notebooks are written to work with this structure, so anyone supplying a similarly formatted CSV can reproduce the analysis.

---

### Step-by-step project flow

Below is a concise, numbered flow showing how the work progresses from raw data to a selected model.

1. **Exploratory Data Analysis (EDA)**  
* Inspect data types and distributions, check missing values, duplicates, linear relationship and class balance.  
* Confirmed problem type: supervised binary classification.

2. **Preprocessing & Pipeline development**  
* Built a sklearn.Pipeline with small, composable custom transformers to:  
* convert data types to float  
* drop exact duplicate rows  
* apply winsorization (to cap extreme outliers)  
* standardize features using z-score (StandardScaler)  
* The pipeline is saved as preprocessing_pipeline.pkl and the transformed DataFrame is saved as df_transformed.pkl for reproducibility.

3. **Modeling & model selection**  
* Trained and tuned multiple classifiers using grid search + Leave-One-Out Cross Validation (LOOCV):  
* Decision Tree  
* Random Forest  
* Logistic Regression  
* XGBoost  
* LightGBM  
* Evaluated models using Accuracy, Precision, Recall, F1, and AUC.  
* Compared models and selected the best performing candidate(The Random Forest).

4. **Further Development and Evaluated of the best performing candidate**

5. **Application of feature selection(SelectKBest)**

6. **Extraction of the feature importance**
7. **Reporting**  
   
---

## Technical details & explanations

**Why winsorize?**  
* Winsorization reduces the influence of extreme outliers by capping values at a defined percentile. In small survey datasets this helps models avoid being skewed by a few extreme responses.

**Why z-score standardization?**  
* Standardization (z-score) centers features to mean 0 and unit variance which helps distance-based and regularized models converge and perform consistently across scales.

**Why LOOCV + GridSearch and 10-Fold stratifiedCV?**  
* With a relatively small dataset (110 samples), Leave-One-Out Cross Validation and 10-Fold StratifiedCV are robust ways to estimate out-of-sample performance while GridSearchCV enables a disciplined hyperparameter search across a small parameter grid.
---

## How to Reproduce Locally (Quick Start)

* These notebooks were developed with Python 3.10 and Jupyter. Paths inside the notebooks may need adjusting depending on your setup.

 *Clone the repository

* git clone https://github.com/Eze-jonas/ivFuindSYvzqwbHB.git
* cd ivFuindSYvzqwbHB


* (Recommended) Create a virtual environment and install dependencies

* python -m venv .venv
* source .venv/bin/activate    # macOS / Linux
* .\.venv\Scripts\activate     # Windows

* pip install --upgrade pip
* pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm scipy


### **Dataset setup**

* The project originally used an internal company survey dataset (ACME-HappinessSurvey2020.csv).

* For confidentiality reasons, the dataset is not included in this repo.

* To run the notebooks yourself, provide a dataset with the same structure:

* Columns: **Y, X1, X2, X3, X4, X5, X6**

* **Y** is the binary target (customer happy or not).

* Update the file_path variable inside the notebooks to point to your dataset.

**Execution order**

* Run Happy Customer Pipeline Dev.ipynb → generates preprocessing artifacts (preprocessing_pipeline.pkl, df_transformed.pkl).

* Run Happy Customer Model Training, Selection and Development.ipynb → performs model tuning, evaluation, feature importance, and feature selection.

* (Optional) Open Happy Customer EDA.ipynb → for exploratory analysis and descriptive insights.

## Repository structure:
* ivFuindSYvzqwbHB/
* Happy Customer EDA.ipynb
* Happy Customer Model Training, Selection and Development.ipynb
* Happy Customer Pipeline Dev.ipynb
* README.md  (this file)

### **What I Contributed — Executive Summary**
1. End-to-end ML workflow ownership: Performed EDA, designed and implemented a reusable preprocessing pipeline, applied feature selection, trained models, tuned hyperparameters, and produced reproducible artifacts.
3. Production-aware design: Built custom scikit-learn transformers and pipelines that can be serialized (preprocessing_pipeline.pkl) and integrated into downstream systems.
4. Rigorous evaluation: Applied LOOCV (suited to small datasets) and stratified 10-fold CV, reporting multiple metrics (Accuracy, Precision, Recall, F1, AUC) for defensible model comparisons.
5. Results-driven experimentation: Identified key survey features driving customer happiness and demonstrated that reduced feature sets can retain predictive performance.
6. Reproducibility & documentation: Delivered well-structured, annotated notebooks so reviewers can step through decisions, validate methodology, and reproduce results.
---
### Suggestions/Recommendations.  
* While the optimistic model - Acc 0.836, Prec: 0.828, Rec: 0.857, F1: 0.842, AUC: 0.926 trained and evaluated with Full dataset is excellent for
feature importance extraction, the improved model (10-Fold Stratified CV evaluation – Acc: 0.655, Prec: 0.655, Rec: 0.679, F1: 0.667, AUC: 0.646 at K=3) shows promise for feature selection. However, it struggles with generalization. An AUC of 0.646 is only slightly above random guessing (0.5), making the model unsuitable for deployment. Despite experimenting with different algorithms, tuning hyperparameters using LOOCV with GridSearch and HyperOpt, applying 10-fold stratified CV within GridSearch, and using feature selection methods (SelectKBest), the dataset remains a key limitation. Future work should focus on increasing the dataset size, since the current 110 rows are too few for reliable deployment."
* The select__k = 3 was chosen, highlighting features X1, X3, and X5. These represent the minimal set of features that preserve accuracy while improving the model’s predictive ability. Future surveys may omit features X2, X4, and X6, as their exclusion does not lead to a loss of predictive information. 
* For every business to grow and develop, customers are needed and to maintain them, they should always be happy trasancting with you. To achieve this happiness focusing on the first 4 feature importance,
1. Ensure that the goods ordering app displays the time at which delivery would be made and esure it is delivered before or at the  sepecified time. This will make customers know when their order will arrive and would not be like they waited too long. If there is any delay, communicate them immediately.
2. Ensure that you employ well mannered couriers and improve their manners by conducting time to time customer/curier relationship training and always let them understand how important a customer is to the success of the business.
3. Conducting sales analysis to find out the most sold products is necessary. This will ensure that these products are always in stock as it will prevent customers checking elsewhere which might make them discover a better prices or services and deside to chun.
4. Pickers should be provided with order list and ensure that they are properly checked and completed befor packing for delivery.



```python

```
