# 📊 Social Media Sentiment Analysis Dashboard

This project is a Streamlit web application that performs sentiment analysis on social media text. It allows users to train machine learning models (Logistic Regression and Naive Bayes) on a labeled dataset, evaluate their performance, and then use the trained models for both live predictions and batch processing of new, unlabeled posts.

The entire application is self-contained in `app.py`.

## ✨ Features

- **Interactive Dashboard**: A user-friendly web interface built with Streamlit.
- **Data Preprocessing**:
    - Cleans text data by removing URLs, HTML tags, mentions, and hashtags.
    - Maps a wide variety of emotion labels into three distinct classes: **Positive**, **Negative**, and **Neutral**.
- **Model Training**:
    - Trains two classic NLP models: **Logistic Regression** and **Multinomial Naive Bayes**.
    - Uses `TF-IDF Vectorizer` for text feature extraction.
    - Allows users to select which models to train and adjust the train-test split ratio.
- **Model Evaluation**:
    - Displays performance metrics including test accuracy and cross-validation accuracy.
    - Shows a detailed classification report (precision, recall, F1-score) for each class.
    - Visualizes a confusion matrix for each trained model.
- **Live Prediction**: A text box for real-time sentiment prediction on any custom text.
- **Batch Processing**:
    - Loads an unlabeled CSV file (`socialmedia.csv`).
    - Runs sentiment prediction on all posts.
    - Provides a download button to save the results as a new CSV file.

## ⚙️ Setup and Installation

To run this application locally, follow these steps.

### 1. Prerequisites

- Python 3.8+
- `pip` for package management

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd DS620-Sentiment-Analysis
```

### 3. Create a Virtual Environment (Recommended)

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

Create a file named `requirements.txt` with the following content:

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
```

Then, install the packages using pip:

```bash
pip install -r requirements.txt
```

## 📁 Required Data Files

Place the following two CSV files in the root directory of the project. The application allows you to specify their paths in the UI sidebar.

1.  **`sentimentdataset.csv`** (for training)
    - This file must contain labeled data.
    - **Required Columns**: `Text` and `Sentiment`.

2.  **`socialmedia.csv`** (for batch prediction)
    - This file contains the unlabeled posts to be analyzed.
    - **Required Columns**: `User ID`, `Username`, `Platform`, `Post ID`, `Post Text`.

## 🚀 How to Run the Application

1.  Make sure your terminal is in the project's root directory and your virtual environment is activated.

2.  Run the Streamlit app with the following command:

    ```bash
    streamlit run app.py
    ```

3.  Your web browser should automatically open a new tab with the application running. If not, navigate to the local URL displayed in your terminal (usually `http://localhost:8501`).

## 📋 How to Use the Dashboard

1.  **Configure Settings**: Use the sidebar on the left to specify the paths to your data files and adjust the test set size.

2.  **Load Training Data**: Click the **"Load & Prepare Training Data"** button. This will load, clean, and display exploratory charts about your training dataset.

3.  **Train Models**: In the "Train Models" section, select the models you wish to train and click the **"Train selected models"** button. The performance metrics and confusion matrices will appear once training is complete.

4.  **Live Prediction**: Go to the "Live & Batch Sentiment Prediction" section. Enter any text into the "Live Sentiment Prediction" box and click **"Predict sentiment"** to see the model's output.

5.  **Batch Scoring**:
    - In the "socialmedia.csv Preview & Batch Scoring" section, click **"Load socialmedia.csv"**.
    - Once loaded, a preview will be displayed.
    - Select a trained model from the dropdown.
    - Click **"Run sentiment on all Post Text and generate file"**.
    - After processing, you can download the results by clicking the **"Download scored socialmedia CSV"** button.