# Parkinson's Disease Classification Using Acoustic Biomarkers and Machine Learning

## Overview

This project investigates the classification of Parkinson's Disease (PD) using replicated acoustic features extracted from voice recordings.

Three supervised machine learning algorithms are implemented and compared:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest (RF)

The project focuses on subject-level evaluation to ensure that recordings belonging to the same individual are not distributed across training and testing sets.

## Dataset

**Dataset:** Parkinson Dataset with Replicated Acoustic Features

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features)

**Dataset DOI:** https://doi.org/10.24432/C5701F

The dataset contains:
- 240 voice recordings
- 80 subjects
- 40 individuals with Parkinson's Disease
- 40 healthy control subjects
- Three repeated recordings per subject

### Acoustic Features

The dataset contains 44 acoustic measurements per recording, including:

- Jitter and Shimmer (relative, absolute, RAP, PPQ, local, dB, APQ3/5/11)
- Harmonic-to-Noise Ratio (HNR) at five frequency bands
- Mel-Frequency Cepstral Coefficients (MFCC0–MFCC12) and their Delta coefficients (Delta0–Delta12)
- Recurrence Period Density Entropy (RPDE)
- Detrended Fluctuation Analysis (DFA)
- Pitch Period Entropy (PPE)
- Glottal-to-Noise Excitation Ratio (GNE)

The target variable is `Status`:
- 0 = Healthy
- 1 = Parkinson's Disease

## Methodology

### 1. Data Preparation
- Loading the dataset using MATLAB.
- Extracting subject IDs, acoustic features, and classification labels.
- Separating features from target labels.

### 2. Subject-Level Train-Test Split
- 80:20 train-test split (192 training recordings, 48 test recordings).
- Splitting performed at the subject level.
- Ensuring recordings from the same subject do not appear in both sets.
- The test set is balanced: 24 healthy and 24 PD recordings.

### 3. Feature Normalization
- Z-score normalization.
- Mean and standard deviation calculated using training data only.
- The same training parameters applied to testing data.

### 4. Machine Learning Models

**K-Nearest Neighbors (KNN)**
- Number of neighbors: 5
- Distance metric: Euclidean

**Support Vector Machine (SVM)**
- Kernel: Radial Basis Function (RBF)
- Kernel scale: Automatic
- Box constraint: 1

**Random Forest (RF)**
- Number of trees: 100
- Classification mode
- All predictors considered at each split.

### 5. Subject-Level Cross-Validation
- Shared 5-fold subject-level cross-validation.
- Identical folds used for all three classifiers.
- Training-only normalization within each fold.

### 6. Performance Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- Receiver Operating Characteristic (ROC) curves
- Area Under the Curve (AUC)

### 7. Feature Importance

Permutation-based feature importance is used to investigate the contribution of individual acoustic features to classification performance.

The top 15 features are identified and visualized for each classifier.

## Results

### Hold-Out Test Set (80:20 Subject-Level Split)

| Metric | KNN | SVM | Random Forest |
|---|---|---|---|
| Accuracy | 75.00% | **83.33%** | 79.17% |
| Precision | 0.9286 | **0.9444** | 0.9375 |
| Recall (Sensitivity) | 0.5417 | **0.7083** | 0.6250 |
| Specificity | 0.9583 | 0.9583 | 0.9583 |
| F1-Score | 0.6842 | **0.8095** | 0.7500 |
| AUC | 0.8385 | 0.8212 | **0.8411** |

**Confusion matrices** (rows = true class, columns = predicted class; 24 recordings per class):

| | KNN | | SVM | | Random Forest | |
|---|---|---|---|---|---|---|
| | Pred. Healthy | Pred. PD | Pred. Healthy | Pred. PD | Pred. Healthy | Pred. PD |
| **True Healthy** | 23 | 1 | 23 | 1 | 23 | 1 |
| **True PD** | 11 | 13 | 7 | 17 | 9 | 15 |

### 5-Fold Subject-Level Cross-Validation (Mean ± Std)

| Metric | KNN | SVM | Random Forest |
|---|---|---|---|
| Accuracy | 75.00 ± 11.22% | **78.75 ± 10.03%** | 75.00 ± 9.32% |
| Precision | 0.7735 ± 0.1242 | **0.8065 ± 0.0835** | 0.7611 ± 0.0862 |
| Recall (Sensitivity) | 0.7083 ± 0.1413 | **0.7500 ± 0.1667** | 0.7250 ± 0.1461 |
| Specificity | 0.7917 ± 0.1179 | **0.8250 ± 0.0618** | 0.7750 ± 0.0757 |
| F1-Score | 0.7367 ± 0.1254 | **0.7731 ± 0.1185** | 0.7393 ± 0.1078 |

### Key Observations

- **SVM achieved the highest accuracy and F1-score** in both the hold-out evaluation (83.33%) and cross-validation (78.75%).
- **Specificity on the hold-out set was identical for all three models (0.9583).** Each misclassified only one healthy recording out of 24, so the differences in accuracy came entirely from sensitivity: SVM detected 17 of 24 PD recordings, RF 15, and KNN 13.
- **AUC values were close** (0.82–0.84). Random Forest had the highest AUC (0.8411) despite SVM having the highest accuracy, which indicates that the models rank samples similarly but differ in the decision threshold at which they classify them.
- **Cross-validated performance was lower than the single hold-out result for SVM and RF**, and fold-to-fold variability was large (standard deviation of roughly 9–11 percentage points in accuracy). The single hold-out split contains only 16 subjects, so its results should be interpreted with caution.
- **Differences between the classifiers are smaller than the fold-to-fold variation** (SVM 78.75% vs. KNN and RF 75.00%, with standard deviations around 10%), so the ranking should not be interpreted as a statistically significant difference.

### Feature Importance (Permutation-Based, Top Features)

| Classifier | Highest-ranked features |
|---|---|
| KNN | GNE, MFCC3, MFCC4, MFCC5, Delta11 |
| SVM | HNR35, PPE, GNE, Delta1 (highest); HNR38, MFCC4, MFCC5, MFCC9, MFCC11, Delta8, Delta11, Delta12 |
| Random Forest | MFCC3, MFCC8, Delta1 (highest); HNR25, HNR35, RPDE, PPE, MFCC0, MFCC2, MFCC10, MFCC12, Delta4 |

- Several features appear in the top rankings of more than one classifier: **GNE** (KNN, SVM), **PPE** (SVM, RF), **HNR35** (SVM, RF), **Delta1** (SVM, RF), **MFCC4 and MFCC5** (KNN, SVM), **Delta11** (KNN, SVM), and **MFCC3** (KNN, RF).
- Importance values are coarse. The permutation importance scores are multiples of 1/48 (≈ 0.0208), corresponding to one test recording, and most features have an importance of zero. The rankings should therefore be treated as indicative rather than definitive.
- Jitter and Shimmer features had zero permutation importance in the top-15 lists for all three classifiers.

### Visual Results

The repository includes:
- Confusion matrices for all three classifiers.
- ROC curves.
- Feature importance plots.
- Comparative performance visualizations.

## Limitations

- The dataset is small (80 subjects), so performance estimates have high variance.
- The hold-out evaluation uses a single split with 16 test subjects.
- Cross-validated AUC was not computed; AUC is reported for the hold-out split only.
- Model hyperparameters were fixed and not tuned.
- No external validation dataset was used, so generalization to other recording conditions or populations is unknown.

## Technologies Used

- MATLAB
- Statistics and Machine Learning Toolbox
- Supervised Machine Learning
- Subject-Level Cross-Validation
- Acoustic Feature Analysis

## How to Run

1. Clone or download this repository.
2. Open MATLAB.
3. Set the project directory as the working directory.
4. Ensure the dataset is available in the expected location.
5. Run `Create_Shared_CV.m` to generate the shared cross-validation folds.
6. Run the individual classification scripts:
   - `KNN_Classification.m`
   - `SVM_Classification.m`
   - `RandomForest_Classification.m`
7. View the generated evaluation metrics and plots.

## Important Note

This project is an experimental machine learning classification study using an existing public dataset. Its predictions are not intended to replace clinical diagnosis or medical assessment.

## Dataset Citation

Prez, C. (2016). *Parkinson Dataset with replicated acoustic features* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5701F.

The dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

## Author

**Ayesha Kauser**

## Acknowledgements

The dataset is provided by the UCI Machine Learning Repository. All dataset ownership and attribution remain with the original creators.
