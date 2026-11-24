# Participant-Independent Classification of VR Induction Conditions Using GSR Signals

A time-series machine learning project investigating whether Galvanic Skin Response (GSR) signals can distinguish between two Virtual Reality (VR) induction conditions: **Nature** and **Subway**.

The project focuses on physiological signal preprocessing, statistical feature extraction, participant-independent validation, and machine learning model evaluation.

> **Research scope:** This project investigates classification of VR induction conditions. It does not claim to directly or objectively recognize human emotional states.

---

## 1. Project Overview

Virtual Reality environments can be designed to induce different emotional experiences. Physiological signals provide a way to investigate how the human body responds to these environments.

This project uses GSR signals collected during two VR conditions:

- **Nature:** A nature-based VR induction condition associated with positive emotional induction.
- **Subway:** An urban subway-based VR induction condition associated with negative emotional induction.

The primary objective is to evaluate whether statistical characteristics and temporal dynamics of GSR signals contain useful information for distinguishing these two conditions.

The experiments emphasize generalization to unseen participants rather than randomly splitting overlapping signal windows.

### Research Objectives

- Inspect and evaluate the quality of the collected GSR recordings.
- Process physiological time-series signals into fixed-length windows.
- Extract statistical and temporal features.
- Investigate the effect of per-recording normalization.
- Compare multiple machine learning classifiers.
- Evaluate participant-independent classification performance.
- Examine differences in classification performance across participants.

---

## 2. Dataset Information

The dataset comes from the research paper:

**Audiovisual Virtual Reality for Emotion Induction: A Dataset of Physiological Responses**

Published in *Scientific Data* (2025).

**Dataset source:**  
https://data.mendeley.com/datasets/y76vbw92y9/3

**Dataset DOI:**  
https://doi.org/10.17632/y76vbw92y9.3

**Research paper:**  
https://www.nature.com/articles/s41597-025-05691-5

### Original Dataset

The original dataset contains physiological recordings from 28 participants, including:

- Electroencephalography (EEG)
- Blood Volume Pulse (BVP)
- Galvanic Skin Response (GSR)
- Skin Temperature (TMP)

This project focuses exclusively on GSR signals to maintain a manageable and focused experimental scope.

### Dataset Subset Used

| Description | Count |
|---|---:|
| Nature recordings | 28 |
| Subway recordings | 18 |
| Total original GSR recordings | 46 |
| Matched participants | 18 |
| Participants included in final experiments | 17 |
| Total extracted windows | 672 |

Nature and Subway recordings were paired using participant identifiers.

Nature participant S3 was excluded from the paired experiments because its recording contained only 32 samples, approximately 7.75 seconds, with a constant GSR value and zero standard deviation.

The remaining 17 matched participants were included in the experiments.

The original recordings were preserved during analysis.

---

## 3. Data Inspection and Quality Assessment

Before feature extraction, the GSR recordings were inspected for basic data-quality issues.

The inspection covered:

- Missing timestamps
- Missing GSR values
- Duplicate timestamps
- Sampling intervals
- Recording durations
- Signal variability
- Basic signal distribution characteristics

### Data Quality Summary

| Metric | Nature | Subway |
|---|---:|---:|
| Files inspected | 28 | 18 |
| Total samples | 36,800 | 20,798 |
| Missing timestamps | 0 | 0 |
| Missing GSR values | 0 | 0 |
| Duplicate timestamps | 0 | 0 |
| Median sampling interval | 0.25 sec | 0.25 sec |

The recordings were sampled at approximately **4 Hz**, corresponding to one sample every 0.25 seconds.

The data inspection helped identify the degenerate Nature S3 recording before the paired classification experiments.

---

## 4. Data Preprocessing

### 4.1 Per-Recording Normalization

The original GSR signals exhibited substantial differences in magnitude across participants and recording conditions.

To reduce the influence of recording-specific signal levels, median-IQR normalization was investigated.

For each complete recording:

\[
GSR_{normalized} =
\frac{GSR - Median(GSR)}{IQR(GSR)}
\]

Where:

- Median represents the recording's median GSR value.
- IQR represents the interquartile range.
- The normalized signal represents deviations relative to the recording's own distribution.

This transformation reduces the influence of differences in absolute signal magnitude and emphasizes relative signal dynamics.

**Important methodological consideration:** Normalization was performed using the complete individual recording before segmentation. Therefore, this experiment represents an offline analysis setting. It should not be interpreted as a real-time deployment pipeline because future samples from a complete recording would not necessarily be available during live prediction.

### 4.2 Window Segmentation

The recordings were divided into fixed-length overlapping windows.

| Parameter | Value |
|---|---:|
| Sampling frequency | 4 Hz |
| Window duration | 30 seconds |
| Samples per window | 120 |
| Overlap | 50% |
| Stride duration | 15 seconds |
| Stride samples | 60 |

Each window contained 120 consecutive GSR samples.

A 50% overlap was applied to increase the number of available temporal segments while preserving local signal information.

### 4.3 Extracted Windows

After preprocessing and segmentation:

| Condition | Windows |
|---|---:|
| Nature | 371 |
| Subway | 301 |
| **Total** | **672** |

The final experimental dataset contained 672 windows from 17 participants.

---

## 5. Feature Extraction

Fourteen statistical and temporal features were initially extracted from each GSR window.

### Extracted Features

| No. | Feature | Description |
|---:|---|---|
| 1 | Mean | Average signal value |
| 2 | Median | Middle signal value |
| 3 | Standard Deviation | Signal variability |
| 4 | Variance | Squared signal variability |
| 5 | Minimum | Minimum signal value |
| 6 | Maximum | Maximum signal value |
| 7 | Range | Difference between maximum and minimum |
| 8 | Q25 | First quartile |
| 9 | Q75 | Third quartile |
| 10 | IQR | Interquartile range |
| 11 | RMS | Root mean square |
| 12 | Slope | Linear trend within the window |
| 13 | Mean Absolute Difference | Average absolute difference between consecutive samples |
| 14 | Difference Standard Deviation | Variability of consecutive-sample differences |

The extracted feature matrix contained:

- 672 observations
- 14 numerical features
- Participant identifiers
- Condition labels

### Feature Selection

A reduced feature set containing six signal-dynamics features was investigated.

The selected features were:

1. Standard Deviation
2. Range
3. IQR
4. Slope
5. Mean Absolute Difference
6. Difference Standard Deviation

The six-feature representation was used in the best-performing experiment.

---

## 6. Machine Learning Experiments

Three classifiers were evaluated:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

Feature scaling was incorporated into machine learning pipelines using `StandardScaler`.

This ensured that scaling parameters were learned from training participants within each validation fold rather than from the entire dataset.

### Experiment 1: Original GSR Features

The first experiment used the original 14 statistical features without per-recording median-IQR normalization.

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 41.96% | 44.84% | 38.75% |
| SVM (RBF) | 42.11% | 44.16% | 40.78% |
| Random Forest | 33.93% | 34.71% | 33.83% |

The initial results indicated limited classification performance when using the original signal features.

### Experiment 2: Normalized GSR Features

The second experiment applied per-recording median-IQR normalization before segmentation and feature extraction.

All 14 features were retained.

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 54.17% | 55.42% | 53.99% |
| SVM (RBF) | 45.83% | 48.12% | 44.23% |
| Random Forest | 47.62% | 47.73% | 47.54% |

Logistic Regression achieved the best performance among the evaluated classifiers.

The results suggested that normalization improved the usefulness of the extracted features for distinguishing the two VR conditions.

### Experiment 3: Selected Six Dynamics Features

The third experiment used the six selected signal-dynamics features from the normalized dataset.

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | **55.95%** | **57.29%** | **55.75%** |
| Random Forest | 47.02% | 47.01% | 46.89% |
| SVM (RBF) | 44.49% | 46.85% | 42.70% |

Logistic Regression with the six selected features achieved the best overall balanced accuracy and macro F1 among the evaluated experiments.

This configuration was retained as the best-performing model.

### Experiment 4: Participant-Balanced Logistic Regression

An additional experiment investigated participant-balanced sample weighting.

The training samples were weighted inversely according to the number of windows contributed by each training participant.

This was intended to reduce the influence of participants contributing more windows to the training dataset.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 56.55% |
| Balanced Accuracy | 56.54% |
| Macro F1 | 56.39% |

Although accuracy and macro F1 increased slightly, balanced accuracy decreased compared with the best six-feature Logistic Regression experiment.

Therefore, participant-balanced weighting was not selected as the final best-performing configuration.

---

## 7. Validation Strategy

### Leave-One-Group-Out Cross-Validation (LOGO)

A participant-independent validation strategy was used.

The dataset was divided according to participant identity using `LeaveOneGroupOut`.

In each fold:

- 16 participants were used for training.
- One unseen participant was used for testing.
- The process was repeated until all 17 participants had served as the test participant.

This resulted in 17 validation folds.

### Why Participant-Independent Validation?

The dataset contains overlapping windows extracted from the same recordings.

A conventional random train-test split could place highly related windows from the same participant in both training and testing sets.

This could produce overly optimistic performance estimates.

Grouping by participant provides a more demanding evaluation of generalization to previously unseen individuals.

All feature scaling was performed within the training pipeline to avoid leakage from test participants.

---

## 8. Final Model Performance

**Best configuration:** Logistic Regression using six selected features from per-recording normalized GSR signals.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 55.95% |
| Balanced Accuracy | 57.29% |
| Macro F1 | 55.75% |
| Macro Precision | 57.65% |
| Macro Recall | 57.29% |

Validation method: 17-fold Leave-One-Group-Out Cross-Validation.

### Final Confusion Matrix

Rows represent actual conditions, while columns represent predicted conditions.

| Actual / Predicted | Nature | Subway |
|---|---:|---:|
| Nature | 165 | 206 |
| Subway | 90 | 211 |

### Class-Specific Recall

| Condition | Correct Predictions | Total Windows | Recall |
|---|---:|---:|---:|
| Nature | 165 | 371 | 44.47% |
| Subway | 211 | 301 | 70.10% |

The model classified Subway windows more successfully than Nature windows.

The confusion matrix also shows that many Nature windows were incorrectly classified as Subway.

These findings indicate that the model's performance is not equally strong across the two conditions.

---

## 9. Participant-Level Analysis

Performance was also examined across individual participants.

The best six-feature Logistic Regression configuration showed considerable variation between participants.

Some participants achieved relatively high balanced accuracy, while others showed very poor classification performance.

Examples:

| Participant | Balanced Accuracy |
|---|---:|
| S12 | 89.61% |
| S14 | 89.33% |
| S8 | 81.94% |
| S15 | 15.79% |
| S5 | 2.78% |
| S4 | 35.71% |

The mean participant-level balanced accuracy was approximately 56.61%.

These differences highlight the challenges of generalizing physiological signal patterns across individuals.

---

## 10. Key Findings

The experiments produced several observations:

1. Original GSR statistical features provided limited participant-independent classification performance.
2. Per-recording median-IQR normalization improved the results of the evaluated classifiers.
3. Reducing the feature set from 14 to six signal-dynamics features improved Logistic Regression performance.
4. Logistic Regression performed better than the evaluated SVM and Random Forest configurations.
5. Participant-balanced sample weighting did not improve balanced accuracy.
6. Classification performance varied considerably across participants.
7. The model showed stronger recall for Subway than Nature.

Overall, the results suggest that GSR signal dynamics contain some information useful for distinguishing the two VR induction conditions, but the observed generalization performance remains modest.

---

## 11. Limitations

This project has several limitations that should be considered when interpreting its findings.

- **Limited participant population:** The final paired experiments included only 17 participants.
- **Participant variability:** Classification performance differed substantially across individuals.
- **Class imbalance:** Nature and Subway contributed different numbers of windows.
- **Modest performance:** The best balanced accuracy was 57.29%, indicating limited discrimination.
- **Condition labels:** The target labels represent VR induction conditions and should not be treated as direct measurements of participants' emotional states.
- **Offline normalization:** Per-recording normalization uses the complete recording, which limits direct applicability to real-time prediction.
- **Single physiological modality:** Only GSR was used, although the original dataset contains additional physiological signals.
- **Feature-based approach:** The experiments focused on handcrafted statistical and temporal features rather than deep learning architectures.

Further research with larger participant populations, additional physiological modalities, and carefully designed real-time preprocessing would be necessary to investigate generalization more thoroughly.

---

## 12. Repository Structure

```text
Project_02_VR_Emotion_Recognition/
│
├── GSR_dataset/                 # Original dataset (not included in GitHub)
│   ├── Nature/
│   └── Subway/
│
├── notebooks/
│   └── 01_VR_GSR_Data_Inspection.ipynb
│
├── .gitignore
│
└── README.md
```

The original dataset directory is excluded from version control because of its size and licensing considerations.

Users should obtain the dataset directly from its official Mendeley Data source.

---

## 13. Tools and Technologies

- **Programming Language:** Python
- **Development Environment:** Jupyter Notebook
- **Data Processing:** Pandas, NumPy
- **Scientific Computing:** SciPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Validation:** Leave-One-Group-Out Cross-Validation

---

## 14. Reproducing the Experiments

### Step 1: Obtain the Dataset

Download the GSR dataset from:

https://data.mendeley.com/datasets/y76vbw92y9/3

Extract the Nature and Subway GSR recordings into the `GSR_dataset` directory.

### Step 2: Set Up Python

Python 3.11 was used for the experimental environment.

Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

### Step 3: Open the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/01_VR_GSR_Data_Inspection.ipynb
```

### Step 4: Configure the Dataset Path

Update the dataset path in the notebook to match the local directory.

### Step 5: Execute the Experiments

Run the notebook cells sequentially to reproduce data inspection, preprocessing, feature extraction, model training, validation, and evaluation.

---

## 15. Conclusion

This project investigated participant-independent classification of two VR induction conditions using GSR time-series signals.

Through data inspection, per-recording normalization, overlapping-window segmentation, statistical feature extraction, and participant-grouped validation, multiple machine learning configurations were evaluated.

The best observed configuration was Logistic Regression using six selected dynamics features, achieving:

- **55.95% Accuracy**
- **57.29% Balanced Accuracy**
- **55.75% Macro F1**

The results demonstrate that GSR signal dynamics may provide useful information for distinguishing VR induction conditions. However, the modest classification performance and substantial participant-level variation indicate that further investigation is necessary before drawing conclusions about reliable generalization.

---

## Author

**Mohammad Deen**  
Computer Science Graduate  
Research Interests: Data Analytics, Machine Learning.

---

## Dataset Citation

Please refer to the original dataset and research publication when using or extending this work:

*Audiovisual Virtual Reality for Emotion Induction: A Dataset of Physiological Responses.* Scientific Data (2025).

Dataset: https://doi.org/10.17632/y76vbw92y9.3

Research paper: https://www.nature.com/articles/s41597-025-05691-5
