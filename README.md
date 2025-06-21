# Credit Card Fraud Detection with ML + Local LLM Explainer

---

## Project Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used is highly unbalanced, with the majority of transactions being legitimate. To address this imbalance, the dataset is under-sampled to create a balanced dataset, enabling the model to detect fraud effectively.

Credit card fraud is a significant issue for financial institutions. The goal of this project is to build a machine learning model that can accurately identify fraudulent transactions. The model is trained using Logistic Regression, a powerful statistical method used for binary classification tasks.

This enhanced version of the project integrates a **locally hosted large language model (TinyLlama via Ollama)** to provide simple, human-friendly explanations of the model’s predictions in a Streamlit web app.

---

## Dataset Information

- **Dataset source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Number of Instances:** 284,807
- **Number of Features:** 30 (anonymized using PCA)
- **Class Labels:**  
  - 0: Legitimate transaction  
  - 1: Fraudulent transaction  
- The dataset is highly unbalanced, with only 492 fraudulent transactions (~0.17% of total).

---

## Features and Implementation

- **Under-Sampling:** To handle class imbalance, legitimate transactions are under-sampled to match the number of fraudulent transactions.
- **Logistic Regression:** Used to train a binary classification model for fraud detection.
- **Data Analysis:** Basic statistical analysis is performed on both legitimate and fraudulent transactions to understand their characteristics.
- **Local LLM Explainer:** Integrates TinyLlama via Ollama to generate easy-to-understand explanations of the model results.
- **Streamlit Web App:** Provides a user-friendly interface to upload data, train the model, and get LLM-generated explanations.

---

## Installation

To run the project locally, ensure you have Python installed.

1. (Optional but recommended) Create and activate a virtual environment:

    - Windows:  
      `python -m venv venv`  
      `venv\Scripts\activate`

    - macOS/Linux:  
      `python -m venv venv`  
      `source venv/bin/activate`

2. Install dependencies:

    ```
    pip install numpy pandas scikit-learn streamlit langchain_community
    ```

3. Install and run Ollama locally by following instructions at [https://ollama.com](https://ollama.com).

4. Download and run the TinyLlama model locally via Ollama:

    ```
    ollama pull tinyllama
    ollama run tinyllama
    ```

---

## Usage

1. Run the Streamlit app:

    ```
    streamlit run main.py
    ```

2. Upload your credit card CSV dataset (make sure it contains the `Class` column).

3. The app will:

    - Display data preview and class distribution.
    - Train a Logistic Regression model on a balanced dataset.
    - Show the accuracy score.
    - Generate a simple explanation using TinyLlama.

---

## Results

- Logistic Regression achieves approximately 95% accuracy on the balanced dataset.
- TinyLlama generates clear, non-technical explanations of the model and key features.

---

## Future Improvements

- Explore other models like Random Forest, XGBoost for better performance.
- Add detailed feature importance metrics and visualizations.
- Enhance the LLM prompts for deeper, more tailored explanations.
- Implement real-time fraud detection with streaming data.
- Deploy the app with a full cloud LLM backend once API costs are manageable.

---

## Contributing

If you would like to contribute to this project, feel free to submit a pull request or raise an issue. Contributions are welcome!

---

## About This Repository

You can either update your existing repo with these new features or create a new repo for this enhanced version — whichever fits your workflow best.

---

## Contact

For any questions or feedback, please open an issue or contact me at [your-email@example.com].

