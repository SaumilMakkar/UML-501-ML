# Resume Category Prediction System - Academic Lab Evaluation Report

---

## Table of Contents

1. [Introduction & Need of the Problem](#1-introduction--need-of-the-problem)
2. [Problem Statement](#2-problem-statement)
3. [Objectives and Existing Approaches](#3-objectives-and-existing-approaches)
4. [Dataset Description](#4-dataset-description)
5. [Methodology / Model Description](#5-methodology--model-description)
6. [Implementation Details](#6-implementation-details)
7. [Results & Evaluation](#7-results--evaluation)
8. [Conclusion](#8-conclusion)
9. [Future Work & Limitations](#9-future-work--limitations)

---

## 1. Introduction

Resume categorization is a critical task in human resource management and recruitment processes. With the increasing volume of job applications received by organizations daily, manual categorization of resumes has become a bottleneck that is time-consuming, error-prone, and difficult to scale. This project addresses the need for an automated system that can accurately classify resumes into appropriate job categories based on their content.

### Need of the Problem

The need for such a system arises from several practical challenges:

- **Time Efficiency**: Manual categorization requires significant human resources and time. HR departments receive hundreds to thousands of resumes daily, making manual screening impractical.

- **Scalability**: As organizations grow, the volume of applications increases exponentially. Manual processes cannot scale effectively.

- **Consistency**: Automated systems eliminate human bias and ensure uniform classification standards across all resumes.

- **Accuracy**: Machine learning models can process and analyze large volumes of text data more accurately than manual screening, reducing misclassification errors.

- **Cost Reduction**: Automating the initial screening process can significantly reduce operational costs and free up HR professionals for more strategic tasks.

This project aims to develop an intelligent system that leverages Natural Language Processing (NLP) and machine learning techniques to automatically categorize resumes into 25 distinct job domains, thereby streamlining the recruitment process and improving efficiency.

**[INSERT DIAGRAM: Recruitment Process Flow - Before and After Automation]**

---

## 2. Problem Statement

The problem addressed in this project is to automatically classify resumes into predefined job categories based solely on the textual content of the resume. The system must:

- Accurately classify resumes into one of 25 job categories
- Handle variations in resume formats, writing styles, and terminology
- Process unstructured text data and extract meaningful features
- Achieve high accuracy rates suitable for production deployment
- Scale efficiently to handle large volumes of resumes

The classification task is a **multiclass text classification problem** where each resume can belong to only one of the 25 categories, requiring the development of robust feature extraction and classification mechanisms.

**Formal Problem Definition**: Given a resume text R, determine the category C ∈ {C₁, C₂, ..., C₂₅} that best describes the candidate's job domain, where Cᵢ represents the i-th job category.

---

## 3. Objectives and Existing Approaches

### 3.1 Objectives

The primary objectives of this project are:

1. **Primary Objective**: To develop an automated resume categorization system with accuracy above 95%

2. **Secondary Objectives**:
   - To implement and compare multiple machine learning algorithms for text classification
   - To evaluate model performance using comprehensive metrics including accuracy, precision, recall, and F1-score
   - To create a production-ready system deployable as a web application
   - To provide confidence scores and top-3 category predictions for transparency
   - To demonstrate the effectiveness of NLP techniques in HR automation

### 3.2 Existing Approaches

Several approaches have been proposed for text classification and resume categorization:

1. **Rule-Based Systems**: Traditional systems using keyword matching and predefined rules. 
   - *Limitations*: Lack of flexibility, poor generalization, requires manual rule creation

2. **Statistical Methods**: Naive Bayes classifiers have been widely used for text classification
   - *Advantages*: Simple, fast, effective with sparse features
   - *Application*: Email spam detection, document classification

3. **Tree-Based Models**: Decision Trees and Random Forests
   - *Advantages*: Interpretable, handle non-linear relationships
   - *Application*: Feature-rich classification tasks

4. **Neural Networks**: Multi-Layer Perceptrons (MLP) and deep learning models
   - *Advantages*: Superior performance in complex text classification tasks
   - *Application*: Large-scale text classification systems

5. **Ensemble Methods**: Combining multiple models through voting or stacking
   - *Advantages*: Improved robustness and accuracy
   - *Application*: Production systems requiring high reliability

This project implements and compares multiple approaches to identify the most effective method for resume categorization.

---

## 4. Dataset Description

### 4.1 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Samples** | 962 resumes |
| **Number of Categories** | 25 job domains |
| **Train-Test Split** | 80% (769 samples) - 20% (193 samples) |
| **Dataset Format** | CSV file |
| **Columns** | Category, Resume (raw text), cleaned_resume |

### 4.2 Category Distribution

The dataset contains resumes from 25 different job categories with varying sample sizes:

**[INSERT CHART: Category Distribution Pie Chart]**

Top 5 Categories:

| Rank | Category | Count | Percentage |
|------|----------|-------|------------|
| 1 | Java Developer | 84 | 8.73% |
| 2 | Testing | 70 | 7.28% |
| 3 | DevOps Engineer | 55 | 5.72% |
| 4 | Python Developer | 48 | 4.99% |
| 5 | Web Designing | 45 | 4.68% |

Complete list of 25 categories: Data Science, HR, Advocate, Arts, Web Designing, Mechanical Engineer, Sales, Health and fitness, Civil Engineer, Java Developer, Business Analyst, SAP Developer, Automation Testing, Electrical Engineering, Operations Manager, Python Developer, DevOps Engineer, Network Security Engineer, PMO, Database, Hadoop, ETL Developer, DotNet Developer, Blockchain, Testing.

### 4.3 Dataset Characteristics

- **Class Distribution**: Imbalanced dataset with categories ranging from 20 to 84 samples
- **Text Length**: Variable resume lengths with rich content including skills, experience, and education
- **Diversity**: Covers both technical and non-technical job categories
- **Quality**: Clean, well-labeled data suitable for supervised learning
- **Format**: Unstructured text data requiring preprocessing

---

## 5. Methodology / Model Description

### 5.1 System Architecture

The methodology follows a systematic machine learning pipeline:

```
Raw Resume Text
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Feature Matrix (7,351 dimensions)
    ↓
Train-Test Split (80-20)
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Best Model Selection
    ↓
Prediction & Deployment
```

**[INSERT DIAGRAM: ML Pipeline Architecture]**

### 5.2 Data Preprocessing

Text preprocessing is performed to clean and normalize resume content:

1. **URL Removal**: Eliminate HTTP/HTTPS links
2. **Hashtag and Mention Removal**: Clean social media artifacts
3. **Punctuation Removal**: Remove special characters
4. **Non-ASCII Character Handling**: Address encoding issues
5. **Whitespace Normalization**: Standardize spacing
6. **Lowercase Conversion**: Normalize text case

**[INSERT DIAGRAM: Data Preprocessing Pipeline Flowchart]**

### 5.3 Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization is used to convert text into numerical features. This technique assigns weights to words based on their frequency in a document relative to their frequency across all documents.

**Formula**: `TF-IDF(t,d) = TF(t,d) × IDF(t)`

Where:
- TF(t,d) = Term Frequency in document
- IDF(t) = Inverse Document Frequency = log(N/df(t))

**Parameters**:
- Features Generated: 7,351 unique features
- Sublinear TF scaling: Enabled
- Stop words: English stop words removed
- Maximum features: Auto-determined

### 5.4 Models Implemented

Six different machine learning algorithms were implemented and compared:

#### 5.4.1 K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning algorithm
- **Strategy**: Classifies based on similarity to k nearest neighbors
- **Wrapper**: OneVsRestClassifier (for multiclass)

#### 5.4.2 Gaussian Naive Bayes
- **Type**: Probabilistic classifier
- **Assumption**: Features follow Gaussian distribution
- **Advantages**: Fast, efficient for text classification
- **Wrapper**: OneVsRestClassifier

#### 5.4.3 Decision Tree
- **Type**: Tree-based model
- **Strategy**: Splits data based on feature values
- **Advantages**: Interpretable, handles non-linear relationships
- **Wrapper**: OneVsRestClassifier

#### 5.4.4 Random Forest
- **Type**: Ensemble of decision trees
- **Advantages**: Robust, reduces overfitting, provides feature importance
- **Wrapper**: OneVsRestClassifier

#### 5.4.5 Logistic Regression
- **Type**: Linear classifier
- **Advantages**: Fast, interpretable, probabilistic outputs
- **Parameters**: Max iterations = 1000
- **Wrapper**: OneVsRestClassifier

#### 5.4.6 MLP Neural Network (Multi-Layer Perceptron)
- **Type**: Deep learning classifier with hidden layers
- **Architecture**: Multi-layer feedforward neural network
- **Parameters**: 
  - Alpha (L2 regularization): 1
  - Max iterations: 1000
- **Advantages**: Non-linear, high capacity, best performance

---

## 6. Implementation Details

### 6.1 Technology Stack

| Component | Technology |
|-----------|-----------|
| Programming Language | Python 3.x |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| NLP | TF-IDF Vectorizer |
| Visualization | matplotlib, seaborn |
| Web Framework | Streamlit |
| Development Environment | Jupyter Notebook |

### 6.2 Training Configuration

- **Data Split**: 80% training (769 samples), 20% testing (193 samples)
- **Stratification**: Enabled to maintain category distribution
- **Random State**: 42 (for reproducibility)
- **One-vs-Rest Strategy**: Applied for multiclass classification
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameters**: Default scikit-learn parameters used

### 6.3 Evaluation Metrics

Models are evaluated using multiple metrics:

- **Accuracy**: Percentage of correctly classified resumes
  - Formula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- **Precision**: Proportion of predicted positives that are actually positive
  - Formula: `Precision = TP / (TP + FP)`

- **Recall**: Proportion of actual positives correctly identified
  - Formula: `Recall = TP / (TP + FN)`

- **F1-Score**: Harmonic mean of precision and recall
  - Formula: `F1-Score = 2 × (Precision × Recall) / (Precision + Recall)`

---

## 7. Results & Evaluation

### 7.1 Model Performance Comparison

All six models were trained and evaluated on the test set. Performance results:

**[INSERT CHART: Model Accuracy Comparison Bar Chart]**

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------|--------|----------|
| **MLP Neural Network** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| Gaussian Naive Bayes | 99.48% | 0.9953 | 0.9948 | 0.9948 |
| Decision Tree | 99.48% | 0.9954 | 0.9948 | 0.9948 |
| Random Forest | 99.48% | 0.9954 | 0.9948 | 0.9948 |
| Logistic Regression | 98.96% | 0.9909 | 0.9896 | 0.9896 |
| K-Nearest Neighbors | 97.93% | 0.9796 | 0.9793 | 0.9785 |

### 7.2 Best Performing Model

**MLP Neural Network** achieved perfect classification with **100% accuracy** on the test set, demonstrating exceptional performance. The model achieved perfect scores across all metrics:
- Precision: 1.0000 (No false positives)
- Recall: 1.0000 (No false negatives)
- F1-Score: 1.0000 (Perfect balance)

**[INSERT CHART: Comprehensive Metrics Comparison]**

### 7.3 Confusion Matrix Analysis

Confusion matrices were generated for all models to analyze classification errors.

**[INSERT CHART: Confusion Matrix Heatmaps for All 6 Models (2×3 Grid)]**

Key observations:
- MLP Neural Network showed perfect classification with all predictions on the diagonal
- Other models showed minimal misclassifications
- Clear patterns in error types for each model

### 7.4 Cross-Validation Results

5-fold cross-validation was performed to assess model stability and generalization.

**[INSERT CHART: Cross-Validation Box Plot]**

Results showed:
- Consistent performance across folds with minimal variance
- Robust models suitable for production deployment
- MLP Neural Network maintained high accuracy across all folds

### 7.5 Key Findings

1. **MLP Neural Network Excellence**: Achieved perfect 100% accuracy, demonstrating the power of deep learning for text classification

2. **Near-Perfect Performance**: Multiple models achieved 99%+ accuracy, indicating high-quality feature engineering

3. **TF-IDF Effectiveness**: The 7,351-dimensional feature space captured sufficient information for accurate classification

4. **Consistent Performance**: All models performed exceptionally well (>97%), suggesting well-structured dataset and clear category distinctions

---

## 8. Conclusion

This project successfully developed an automated resume categorization system that achieves exceptional performance across multiple machine learning algorithms. The MLP Neural Network emerged as the best performing model with **100% accuracy** on the test set.

### Key Achievements

- ✅ Successfully classified 962 resumes into 25 job categories with high accuracy
- ✅ Implemented and compared 6 different ML algorithms, providing comprehensive evaluation
- ✅ Achieved perfect classification (100%) with MLP Neural Network
- ✅ Developed a production-ready system deployable as a web application
- ✅ Demonstrated the effectiveness of NLP and ML techniques in HR automation

### Impact

The results demonstrate that automated resume categorization is feasible and can significantly improve efficiency in HR processes. The high accuracy rates achieved suggest that the system is suitable for real-world deployment and can provide substantial time savings in resume screening processes.

### Technical Contributions

- Demonstrated effectiveness of MLP for text classification
- Showed value of TF-IDF feature engineering approach
- Established benchmark performance for resume classification
- Validated ensemble methods for production systems

---

## 9. Future Work & Limitations

### 9.1 Limitations

While the system achieved excellent results, several limitations exist:

1. **Dataset Size**: The dataset contains 962 samples, which may limit generalization to diverse resume styles

2. **Class Imbalance**: Uneven distribution across categories (20-84 samples) may affect minority class performance

3. **Language**: System is currently limited to English-language resumes

4. **Format Dependency**: Performance may vary with different resume formats and structures

5. **Feature Engineering**: Current approach uses TF-IDF; advanced embeddings (Word2Vec, BERT) were not explored

6. **Hyperparameter Tuning**: Default parameters were used; optimization could improve performance further

### 9.2 Future Work

Future enhancements and research directions include:

1. **Dataset Expansion**:
   - Increase dataset size and diversity to improve generalization
   - Add more job categories
   - Include resumes from different industries and regions

2. **Advanced Models**:
   - Explore Transformer models (BERT, GPT) for enhanced text understanding
   - Implement attention mechanisms
   - Experiment with pre-trained language models

3. **Feature Engineering**:
   - Implement semantic embeddings (Word2Vec, GloVe)
   - Extract named entities (skills, certifications, degrees)
   - Add temporal features (years of experience, date patterns)

4. **System Enhancements**:
   - Multi-language support for non-English resumes
   - Real-time processing optimization
   - Improve PDF parsing and OCR capabilities
   - Provide confidence intervals and uncertainty estimates

5. **Imbalance Handling**:
   - Apply techniques like SMOTE or class weighting
   - Use ensemble methods with balanced sampling

6. **Deployment Improvements**:
   - RESTful API development for integration
   - Batch processing capabilities
   - Dashboard analytics
   - Skill gap analysis and recommendations

---

## References

1. Scikit-learn Documentation. Available at: https://scikit-learn.org/
2. Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, pp. 2825-2830, 2011.
3. Manning, C., et al. "Introduction to Information Retrieval." Cambridge University Press, 2008.
4. Streamlit Documentation. Available at: https://docs.streamlit.io/

---

**Report Generated**: [Date]  
**Author**: [Your Name]  
**Course**: Machine Learning  
**Institution**: [Institution Name]

---
