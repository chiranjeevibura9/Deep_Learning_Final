# 🧠 Deep Learning Final Project: Text Classification

## 📌 Project Overview
This project focuses on **text classification using deep learning**, where we aim to categorize text documents into **20 different newsgroups** based on their content.

### **Dataset: 20 Newsgroups**
- **Contains**: ~20,000 text documents across **20 categories**.
- **Task**: Classify text documents into their respective **newsgroup categories** using **deep learning architectures**.

### **Goal**
- Explore different **neural architectures** (LSTMs, CNNs, Transformer-based models).
- Analyze **model performance and optimization techniques**.
- Compare traditional deep learning models with **state-of-the-art transformers**.

---

## 📂 Table of Contents
1. [Introduction](#introduction)
2. [Problem Identification](#problem-identification)
3. [Data Collection and Provenance](#data-collection-and-provenance)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Deep Learning Model Training](#deep-learning-model-training)
6. [Results and Discussion](#results-and-discussion)
7. [Conclusion](#conclusion)
8. [Installation and Usage](#installation-and-usage)
9. [Future Enhancements](#future-enhancements)
10. [Repository Structure](#repository-structure)
11. [License](#license)

---

## 🔬 Introduction
- This project implements **deep learning models** for **automated text classification**.
- We compare the performance of **CNN, LSTM, Bi-LSTM, and Transformer-based architectures**.
- The dataset used is **20 Newsgroups**, which contains **news articles across 20 categories**.

---

## 📥 Data Collection and Provenance
- The dataset is sourced from **[Scikit-Learn's 20 Newsgroups Dataset](https://scikit-learn.org/stable/datasets/twenty_newsgroups.html)**.
- No additional web scraping or external modifications were performed.
- The dataset is split into **training (60%) and test (40%) sets**.

---

## 📊 Exploratory Data Analysis (EDA)
- **Tokenization & Text Preprocessing**:
  ✔ Remove **stopwords, punctuation, and special characters**  
  ✔ Convert text to **lowercase**  
  ✔ Apply **TF-IDF vectorization & Word Embeddings (FastText, GloVe)**  

- **Category Distribution**:
  - Balanced distribution across **20 categories**.
  - Visualized using **word clouds & bar plots**.

---

## ⚙ Deep Learning Model Training
### **Models Implemented:**
✅ **CNN for Text Classification**  
✅ **LSTM & Bi-LSTM for Sequential Text Processing**  
✅ **Transformer-based models (BERT, DistilBERT, RoBERTa)**  

### **Evaluation Metrics:**
- **Accuracy** ✔  
- **Precision, Recall, F1-Score** ✔  
- **Confusion Matrix & ROC Curve** ✔  

### **Hyperparameter Tuning Techniques Used**
🔹 **Learning Rate Scheduling** (ReduceLROnPlateau)  
🔹 **Batch Size & Dropout Optimization**  
🔹 **Early Stopping & Regularization**  

---

## 📈 Results and Discussion
### **Model Performance Overview**
| Model | Accuracy | F1-Score |
|--------|---------|----------|
| CNN | 87.2% | 0.86 |
| LSTM | 89.5% | 0.89 |
| Bi-LSTM | 90.1% | 0.90 |
| **BERT** | **95.7%** | **0.95** |

✅ **BERT achieved the highest accuracy (95.7%)**, outperforming traditional deep learning models.  
✅ **LSTM/Bi-LSTM models** performed well but were **slower** in training compared to **CNNs**.  
✅ **CNNs** were **faster but less accurate** for longer text sequences.  

---

## 🛠 Installation and Usage
### **Prerequisites**
- Python 3.8+
- Jupyter Notebook
- Required Python libraries:
  ```bash
  pip install numpy pandas matplotlib scikit-learn tensorflow keras transformers torch
  ```

### **Run the Project**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chiranjeevibura9/Deep_Learning_Final_Project.git
   cd Deep_Learning_Final_Project
   ```

2. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
3. **Open `deep_learning_text_classification.ipynb`** and execute the cells.

---

## 🚀 Future Enhancements
✅ **Train models on larger datasets (Reddit/StackOverflow Corpus)**  
✅ **Experiment with zero-shot learning models (GPT-4, T5)**  
✅ **Fine-tune BERT/RoBERTa for improved domain adaptation**  
✅ **Deploy model as an API using FastAPI or Flask**  

---

## 📁 Repository Structure
```yaml
├── data/
│   ├── train/  # Training dataset
│   ├── test/   # Test dataset
│
├── notebooks/
│   ├── deep_learning_text_classification.ipynb  # Main Jupyter Notebook
│   ├── bert_finetuning.ipynb   # Fine-tuning transformer models
│
├── src/
│   ├── preprocess.py  # Data cleaning scripts
│   ├── train_model.py   # Model training script
│
├── README.md  # Project Documentation
└── requirements.txt  # Required Python libraries
```

---

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing
Contributions are welcome!  
- Fork the repository and submit a **pull request** with improvements.  

---

## 📬 Contact
📧 **Chiranjeevi Bura** - [GitHub](https://github.com/chiranjeevibura9)  
🌐 **Project Repo**: [Deep_Learning_Final_Project](https://github.com/chiranjeevibura9/Deep_Learning_Final_Project)

---

🚀 **If you find this project helpful, give it a ⭐ on GitHub!**  
