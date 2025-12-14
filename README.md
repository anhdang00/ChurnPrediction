# Customer Churn Prediction and Analysis

This project explores churn prediction - predicting which customers are more likely to discontinue their subscription service using the Telco Customer Churn dataset. The analysis compares simple machine learning models with a deep learning approach to identify which provides better predictions and business insights.

## Project Overview

The project analyzes customer demographics, subscribed services, and account information to classify customers as "returned" or "churned". It implements multiple approaches including Logistic Regression, Random Forest, and Deep Learning using TensorFlow/Keras.

## Dependencies

This project requires Python 3.8 or higher. All dependencies are listed in [requirements.txt](requirements.txt).

### Key Libraries

- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Deep Learning**: tensorflow, keras
- **Notebook Environment**: jupyter, jupyterlab

### Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dataset

### Dataset Information

**Source**: Telco Customer Churn dataset from Kaggle

**Location**: `dataset/Telco-Customer-Churn.csv`

**Description**: The dataset contains customer information from a telecommunications company, including:
- Customer demographics - gender, age range, partners, dependents
- Subcribed services - phone, internet, online security, etc.
- Account information - tenure, contract type, payment method, charges
- Churn status - whether the customer left in the last month (0: retained, 1: churned)

### Setting up the Dataset

1. Download the dataset:
    - Visit the Kaggle dataset page for Telco Customer Churn
    - Download `Telco-Customer-Churn.csv`
2. Place the dataset file `Telco-Customer-Churn.csv` in the `dataset/` directory
3. The project should have the following structure:
   ```
   ChurnPrediction/
   ├── Churn_prediction.ipynb
   ├── requirements.txt
   ├── README.md
   └── dataset/
       └── Telco-Customer-Churn.csv
   ```

## How to Run the Code

### Option 1: Using Jupyter Notebook

1. After launching Jupyter notebook or Jupyter lab, navigate to `Churn_prediction.ipynb` in the Jupyter interface and click to open the notebook.
2. Execute cells sequentially from top to bottom.
3. Use `Shift + Enter` to run each cell or use "Run All" from the Cell menu to execute all cells at once.

### Option 2: Using VS Code

1. Open the project folder in VS Code.
2. Install the Jupyter extension if not already installed.
3. Open [Churn_prediction.ipynb](Churn_prediction.ipynb).
4. Select your Python interpreter with dependencies installed.
5. Run cells by clicking the play button or using `Shift + Enter`.

## Reproduction Steps

### Step 1: Set Up the Environment

```bash
# Clone or download the project
cd ChurnPrediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Dataset

Ensure `dataset/Telco-Customer-Churn.csv` exists in the project directory.

### Step 3: Run the Analysis

Open and execute [Churn_prediction.ipynb](Churn_prediction.ipynb) following this sequence:

1. **Import Libraries**: Load all required packages
2. **Load Dataset**: Read the Telco Customer Churn CSV file
3. **Data Preprocessing**: 
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
4. **Feature Engineering**: Prepare features for modeling
5. **Exploratory Data Analysis**:
   - Visualize churn distribution
   - Analyze feature correlations
   - Examine customer demographics
6. **Model Training**:
   - Train Logistic Regression model
   - Train Random Forest classifier
   - Build and train Deep Learning model (TensorFlow/Keras)
7. **Model Evaluation**:
   - Compare accuracy, F1-score, and ROC-AUC metrics
   - Generate classification reports
   - Visualize model performance
8. **Results Analysis**: Interpret findings and draw conclusions

### Step 4: Review Results

- Check model performance metrics
- Compare traditional ML vs Deep Learning approaches
- Review visualizations and insights

## Project Structure

```
ChurnPrediction/
│
├── Churn_prediction.ipynb    # Main Jupyter notebook with complete analysis
├── requirements.txt           # Python package dependencies
├── README.md                  # This file
└── dataset/
    └── Telco-Customer-Churn.csv  # Customer churn dataset
```

## Expected Outputs

The notebook will generate:
- Data visualizations (distribution plots, correlation heatmaps, etc.)
- Model performance metrics (accuracy, F1-score, ROC-AUC)
- Classification reports for each model
- Comparison charts between different models
- Feature importance analysis (for tree-based models)