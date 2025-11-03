# 🎯 Resume Categorizing - AI-Powered Classification System

## Overview
An advanced machine learning project that automatically categorizes resumes into 25 different job domains using state-of-the-art NLP and ML techniques.

## 🚀 Key Features

### 1. **Multi-Model Comparison** 📊
- 6 different algorithms tested and compared
- K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Random Forest, Logistic Regression, and MLP Neural Network

### 2. **Advanced Evaluation** 🔬
- Comprehensive confusion matrices for all models
- Cross-validation analysis (5-fold) for robust evaluation
- Hyperparameter tuning with GridSearchCV
- Multiple metrics: Accuracy, Precision, Recall, F1-Score

### 3. **Ensemble Learning** 🎯
- Voting classifier combining top 4 models
- Improved prediction stability
- Production-ready deployment

### 4. **Real-Time Predictions** 🤖
- Live demo with confidence scores
- Top-3 category suggestions with probability bars
- Interactive prediction function

### 5. **AI-Powered Analytics** 🔍
- Skill extraction per category
- Feature importance analysis
- Category-specific performance insights

### 6. **Professional Visualization** 📈
- Beautiful heatmaps and comparison plots
- Model performance dashboards
- Category distribution analysis

### 7. **Intelligent Job Recommendations** 🎁
- Personalized company recommendations (top 10)
- Best job platforms to apply
- Average salary insights
- Key skills to master

### 8. **Interactive Resume Examples** 📝
- 3 complete resume demonstrations
- Full prediction analysis
- Comprehensive recommendations
- Real-world use cases

## 📁 Project Structure

```
ML-Project/
├── Dataset/
│   ├── UpdatedResumeDataSet.csv    # 962 resume samples
│   └── README.md
├── Model/
│   ├── Resume_categorizing.ipynb   # Main notebook
│   └── README.md
└── requirements.txt
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📊 Dataset

- **Total Samples**: 962 resumes
- **Categories**: 25 job domains
- **Classes**: Data Science, HR, Advocate, Arts, Web Designing, Mechanical Engineer, Sales, Health and fitness, Civil Engineer, Java Developer, Business Analyst, SAP Developer, Automation Testing, Electrical Engineering, Operations Manager, Python Developer, DevOps Engineer, Network Security Engineer, PMO, Database, Hadoop, ETL Developer, DotNet Developer, Blockchain, Testing

## 🎯 Results

### Model Performance (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | **99%+** | 0.99 | 0.99 | 0.99 |
| Random Forest | **99%+** | 0.99 | 0.99 | 0.99 |
| MLP Neural Network | **99%+** | 0.99 | 0.99 | 0.99 |
| Gaussian Naive Bayes | **99%+** | 0.99 | 0.99 | 0.99 |
| Decision Tree | **99%+** | 0.99 | 0.99 | 0.99 |
| K-Nearest Neighbors | **98%+** | 0.98 | 0.98 | 0.98 |

### 🏆 Winner
**Logistic Regression** achieves the highest accuracy and consistency across all evaluation metrics!

## 💡 Key Innovations

1. ✅ **Industry-Ready**: Ensemble voting for production deployment
2. ✅ **Explainable AI**: Confidence scores, feature importance, recommendations
3. ✅ **Robust Evaluation**: CV + tuned hyperparameters + diverse metrics
4. ✅ **Visual Excellence**: Professional plots and interactive dashboards
5. ✅ **Practical Use**: Real-time prediction with actionable insights

## 🔬 Technical Details

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 7,351 dimensions
- **Train/Test Split**: 80/20 with stratification
- **Preprocessing**: URL removal, special characters, whitespace normalization

## 📝 Usage

Run the Jupyter notebook `Resume_categorizing.ipynb` and execute all cells to:
1. Load and preprocess the dataset
2. Train all 6 models
3. Evaluate and compare performance
4. Generate visualizations
5. Run real-time predictions

## 🎓 Authors

Developed as part of an ML course project demonstrating advanced classification techniques.

## 📄 License

Educational project - for academic purposes only.

