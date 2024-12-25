# Credit Card Fraud Detection

## Project Overview
The Credit Card Fraud Detection project aims to build a predictive model to identify fraudulent transactions from a dataset of credit card transactions. This model can assist financial institutions in detecting fraudulent activity and minimizing losses. The dataset is highly imbalanced, containing a vast majority of legitimate transactions compared to fraudulent ones.

---

## Dataset
- **Rows**: 568,630
- **Columns**: 31

### Features
1. **V1, V2, ..., V28**: Principal components obtained through PCA.
2. **Time**: The time elapsed between each transaction and the first transaction in the dataset.
3. **Amount**: The transaction amount.
4. **Class**: Target variable (0: Legitimate, 1: Fraudulent).

---

## Data Preprocessing
1. Addressed class imbalance using techniques such as oversampling (SMOTE) or undersampling.
2. Scaled numerical features (`Time` and `Amount`) using StandardScaler.
3. Split the dataset into training and testing sets (80%-20%).

---

## Exploratory Data Analysis (EDA)
1. Analyzed the distribution of transaction amounts and time.
2. Visualized the imbalance in the target variable.
3. Examined correlations between features using heatmaps.

---

## Modeling
### Algorithms Used
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boosting
- Naive Bayes

### Performance Metrics
- **Accuracy**: Evaluates the overall correctness of the model.
- **Precision**: Measures the proportion of true frauds among predicted frauds.
- **Recall (Sensitivity)**: Measures the proportion of actual frauds detected.
- **F1-Score**: Harmonic mean of precision and recall.
- **AUC-ROC Curve**: Evaluates the ability of the model to distinguish between classes.

---

## Best Model
The **Random Forest** algorithm performed the best with:
- **Accuracy**: 99.98%
- **AUC Score**: 0.98
- **Precision, Recall, and F1-Score**: 0.98 for both classes.

---

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, XGBoost
- **Environment**: Jupyter Notebook / Google Colab

---

## Key Visualizations
1. Class distribution bar plot.
2. PCA components plotted to visualize separability.
3. ROC curve comparison for all models.

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/username/credit-card-fraud-detection.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python scripts provided in the repository.

---

## Challenges
1. Highly imbalanced dataset required advanced sampling techniques.
2. Computational complexity due to the size of the dataset and model tuning.

---

## Future Scope
1. Explore deep learning models such as Autoencoders for anomaly detection.
2. Implement real-time fraud detection systems.
3. Integrate additional features like customer demographics and transaction location.

---

## Contributors
- [Rahul Kumar](https://github.com/rahulkumar-24)

---


