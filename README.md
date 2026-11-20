# Machine Learning Portfolio
 
Selected projects from my graduate Machine Learning and Data Science coursework (University of Miami, MSDS). They cover classical supervised learning on tabular data (logistic regression, decision trees, random forests, KNN, XGBoost), model selection and evaluation on imbalanced data, model explainability with SHAP, geospatial visualization, and deep learning with PyTorch.
 
Everything is written in Python (Jupyter notebooks).
 
## Projects
 
| # | Project | Task | Models | Highlights |
|---|---------|------|--------|------------|
| 1 | [Diabetes Risk Prediction](#1-diabetes-risk-prediction-pima-indians) | Binary classification | Logistic Regression, Random Forest | Recall-first evaluation, ROC/AUC, SHAP |
| 2 | [Credit Default Risk](#2-credit-default-risk-give-me-some-credit) | Binary classification, imbalanced (6.7% positives) | Decision Tree, Random Forest, KNN | Class weights vs. upsampling, validation-based tuning |
| 3 | [House Price Regression (Ames and California)](#3-house-price-regression-ames-iowa-and-california) | Regression | Linear, KNN, Random Forest, XGBoost | 10-fold cross-validation, latitude/longitude map visualization |
| 4 | [ECG Heartbeat Classification](#4-ecg-heartbeat-classification-pytorch) | 5-class time-series classification | MLP (PyTorch), residual 1D CNN (in progress) | LayerNorm, Adam, checkpointing |
| 5 | [Generative Models with PCA and GMM](#5-generative-models-pca-and-gmm) | Dimensionality reduction, generation | PCA, Gaussian Mixture Models | Fashion-MNIST, face images |
| 6 | [Hugging Face Vision Pipelines](#6-hugging-face-vision-pipelines) | Object detection, image classification | DETR, ViT classifier | Pre-trained model inference |
 
---
 
## 1. Diabetes Risk Prediction (Pima Indians)
 
Predict whether a patient has diabetes from 8 clinical measurements (768 patients, 34.9% positive).
 
**Approach**
- Treated physiologically impossible zeros (glucose, blood pressure, skin thickness, insulin, BMI) as missing values and imputed them with the training-set median.
- Stratified 80/20 train/test split. Standardized features for the linear model, fit on the training set only.
- Handled class imbalance with `class_weight='balanced'` and judged models with balanced accuracy, recall and AUC rather than plain accuracy.
- Tuned Random Forest depth with 5-fold `GridSearchCV` (scoring: balanced accuracy), which selected `max_depth=3`.
- Explained both models with SHAP (`TreeExplainer`, `LinearExplainer`).
**Results (test set, 154 patients)**
 
| Model | Accuracy | Balanced acc. | Recall (diabetes) | Precision (diabetes) | AUC |
|-------|---------:|--------------:|------------------:|---------------------:|----:|
| Logistic Regression | 0.766 | 0.756 | 0.72 | 0.65 | **0.874** |
| Random Forest (depth 3) | **0.779** | **0.770** | **0.74** | 0.67 | 0.839 |
 
**Takeaway.** For a screening use case, false negatives (missed diabetics) are the costly error, so recall on the positive class is the primary criterion. Random Forest catches slightly more positives, while logistic regression ranks risk better overall (higher AUC) and is easier to interpret.
 
---
 
## 2. Credit Default Risk (Give Me Some Credit)
 
Predict whether a borrower will experience serious delinquency (90+ days past due) within two years, using 10 credit-history features (150,000 borrowers, from the [Kaggle competition](https://www.kaggle.com/c/GiveMeSomeCredit)).
 
**Approach**
- Median imputation for missing `MonthlyIncome` and `NumberOfDependents`.
- Highly imbalanced data: 139,974 non-defaulters vs. 10,026 defaulters. Split into train / validation / test (108,000 / 12,000 / 30,000).
- Baseline logistic regression, then Decision Tree and Random Forest with `class_weight='balanced'`.
- KNN has no class weights, so the minority class was **upsampled** to balance the training set.
- Chose `max_depth` and `n_neighbors` with a grid search on the validation set, then evaluated once on the test set with confusion matrices.
Because accuracy is misleading when only 6.7% of borrowers default, every model is scored with **balanced accuracy** (the mean of the per-class recalls).
 
**Results (test set, 30,000 borrowers)**
 
| Model | Tuning | Balanced accuracy |
|-------|--------|------------------:|
| Logistic Regression (baseline) | `class_weight='balanced'` | 0.710 |
| Decision Tree | depth 20 (untuned) / **depth 9** (tuned) | 0.668 / 0.742 |
| Random Forest (20 trees) | depth 20 (untuned) / **depth 11** (tuned) | 0.693 / **0.754** |
| KNN | K=5 on raw data / **K=45** on upsampled data | 0.553 / 0.683 |
 
The tuned Random Forest correctly flags 65% of the borrowers who go on to default (1,323 of 2,043) while correctly clearing 86% of the non-defaulters. Tuning depth on the validation set lifted both tree models by 6 to 7 points over their untuned versions, and upsampling lifted KNN by 13 points, but KNN stayed behind the tree models.
 
**Skills shown:** imbalanced classification, resampling vs. class weighting, train/validation/test discipline, hyper-parameter search.
 
---
 
## 3. House Price Regression: Ames (Iowa) and California
 
Two regression studies on residential sale prices, using the same four-model comparison (Linear Regression, KNN, Random Forest, XGBoost) and validation-based hyper-parameter tuning.
 
### 3a. Ames, Iowa home sales (Kaggle *House Prices*)
 
Predict `SalePrice` from property characteristics (1,460 training and 1,459 test homes, 289 features after encoding; [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)).
 
- Dropped features with 50% or more missing values, filled the remaining gaps, and one-hot encoded the categorical variables.
- Used **10-fold cross-validation** (`GridSearchCV` with a custom -MAPE scorer) to choose `n_neighbors`, `max_depth` for Random Forest, and `max_depth` for XGBoost (best: 7 neighbors, depth 10 and depth 6).
| Model | MAE (USD) | RMSE (USD) | MAPE |
|-------|----------:|-----------:|-----:|
| Linear Regression | 11,916 | 21,109 | 7.27% |
| KNN (k=7) | 26,345 | 44,836 | 15.31% |
| Random Forest (10 trees, depth 10) | 12,625 | 25,863 | 6.99% |
| XGBoost (10 trees, depth 6) | 11,929 | 27,168 | **6.81%** |
 
Test set. XGBoost has the lowest MAPE, but all three of Linear, Random Forest and XGBoost land within half a point of each other, and the linear model has the lowest RMSE. After careful feature preparation, a simple linear model was as good as the non-linear ones, and KNN was clearly the weakest. A single very expensive home (about USD 1.1M) is under-predicted by every model, which inflates the squared-error metrics.
 
### 3b. California housing blocks (geospatial)
 
Predict the median house value of a California census block from location and demographic features (20,640 blocks).
 
- Dropped rows with missing `total_bedrooms` (207), one-hot encoded `ocean_proximity`, standardized features.
- Plotted blocks by **latitude/longitude on a map of California**, colored by income and price, to see the spatial structure the models later exploit.
- Selected `max_depth` for Random Forest and XGBoost on a validation set (validation-MAE curve and elbow method); both settled on `max_depth=14`.
- Evaluated with MSE, MAE, MAPE and R², plus predicted-vs-actual plots and prediction maps.
| Model (test set, 4,128 blocks) | MAE (USD) | MAPE |
|-------|----------:|-----:|
| Linear Regression | 49,656 | 28.95% |
| KNN (k=5) | 42,816 | 23.26% |
| Random Forest (20 trees, depth 14) | 33,460 | 19.02% |
| XGBoost (20 trees, depth 14) | **32,200** | **17.95%** |
 
Test R² was 0.64 for the linear model and 0.69 for KNN. Here, location and income interact non-linearly, so the tree-based models clearly beat the linear baseline, in contrast with the Ames data, where a linear model was competitive.
 
**Skills shown:** feature engineering, missing-data handling, cross-validation, geospatial plotting, regression metrics, tree-based models.
 
---
 
## 4. ECG Heartbeat Classification (PyTorch)
 
Classify single heartbeats into 5 classes from raw ECG signals (3,841 samples of 187 time steps; class counts 800 / 800 / 800 / 641 / 800), with fixed train/validation/test splits.
 
**MLP (completed).** 187 → 256 → 256 → 256 → 256 → 256 → 5, ReLU activations, `LayerNorm` before the last three hidden blocks, cross-entropy loss (softmax folded into the loss). Trained with Adam (lr 1e-3, AMSGrad), batch size 128, 100 epochs, keeping the checkpoint with the best validation accuracy.
 
| Metric | Value |
|--------|------:|
| Test accuracy (best-validation checkpoint) | **0.886** |
| Best epoch | 42 |
| Per-class sensitivity | 0.81 / 0.89 / 0.89 / 0.89 / 0.95 |
 
**1D CNN with residual connections (in progress).** Convolution + ReLU stages with average-pooling skip connections and `LayerNorm`, followed by two fully connected layers.
 
**Skills shown:** PyTorch `nn.Module` design, tensor shape reasoning, normalization layers, validation-based checkpointing.
 
---
 
## 5. Generative Models: PCA and GMM
 
- **PCA on Fashion-MNIST:** projected images to 2D for visualization, then sampled in the latent space and reconstructed images to generate new clothing items.
- **GMM + PCA on face images:** a Gaussian Mixture Model is slow in high dimensions, so faces were first compressed with PCA, a GMM was fit in the reduced space, and new faces were generated by sampling and inverting the PCA.
---
 
## 6. Hugging Face Vision Pipelines
 
- **Object detection** with a pre-trained DETR model: filter detections by confidence threshold and draw labeled bounding boxes on the image.
- **Image classification** with a pre-trained Hugging Face `pipeline` (ViT-based content classifier), batch inference over a list of images.
---
 
## Tech stack
 
Python, NumPy, pandas, scikit-learn, XGBoost, SHAP, PyTorch, Hugging Face Transformers, Matplotlib, Seaborn, Jupyter.
 
## Repository layout (suggested)
 
```
.
├── 01-diabetes-risk/          # notebook + figures
├── 02-credit-default-risk/
├── 03-house-price-regression/   # Ames + California
├── 04-ecg-heartbeat-pytorch/  # notebooks + best checkpoints (.pt)
├── 05-pca-gmm-generative/
├── 06-huggingface-vision/
└── README.md
```
 
## Reproducing the results
 
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn jupyter torch transformers
jupyter notebook
```
 
Datasets are not included; each project folder links to or names its source. Python 3.11 or 3.12 is recommended for the PyTorch and Hugging Face notebooks.
 
## Author
Ilan Sasson • https://www.linkedin.com/in/ilansassonb/ • ilansassonbaum15@gmail.com
