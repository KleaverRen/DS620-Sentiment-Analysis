import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# =========================================================
# 1. Label mapping + text cleaning
# =========================================================


def map_to_three_classes(label: str) -> str:
    """
    Map many fine-grained emotion labels into 3 general
    sentiment classes: Positive, Negative, Neutral.
    """
    if label is None or not isinstance(label, str):
        return None

    lab = label.strip().lower()

    # Explicit canonical labels first
    if lab in ["positive"]:
        return "Positive"
    if lab in ["negative"]:
        return "Negative"
    if lab in ["neutral"]:
        return "Neutral"

    positive_keywords = {
        "positive",
        "happy",
        "joy",
        "happiness",
        "enjoyment",
        "amusement",
        "admiration",
        "affection",
        "awe",
        "calmness",
        "excitement",
        "kind",
        "pride",
        "elation",
        "euphoria",
        "contentment",
        "serenity",
        "gratitude",
        "hopeful",
        "hope",
        "empowerment",
        "compassion",
        "tenderness",
        "enthusiasm",
        "fulfillment",
        "reverence",
        "triumph",
        "satisfaction",
        "heartwarming",
        "solace",
        "harmony",
        "ecstasy",
        "magic",
        "beauty",
        "wonder",
        "inspiration",
        "delight",
        "love",
        "charm",
        "engagement",
    }

    negative_keywords = {
        "negative",
        "anger",
        "angry",
        "fear",
        "sadness",
        "sad",
        "disgust",
        "disappointed",
        "bitter",
        "shame",
        "despair",
        "grief",
        "loneliness",
        "jealousy",
        "resentment",
        "anxiety",
        "stress",
        "frustration",
        "boredom",
        "hate",
        "bad",
        "upset",
        "miserable",
        "hurt",
        "worried",
        "regret",
        "guilt",
    }

    if lab in positive_keywords:
        return "Positive"
    if lab in negative_keywords:
        return "Negative"

    if any(pk in lab for pk in positive_keywords):
        return "Positive"
    if any(nk in lab for nk in negative_keywords):
        return "Negative"

    # Treat all other labels as Neutral
    return "Neutral"


URL_PATTERN = re.compile(r"http\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
HTML_TAG_PATTERN = re.compile(r"<.*?>")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# 2. Data loading (cached)
# =========================================================


@st.cache_data
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load labeled dataset (e.g., sentimentdataset.csv) used for training.
    Needs columns: Text, Sentiment
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if "Text" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError(
            "Expected columns 'Text' and 'Sentiment'. "
            "Please adjust the code or your CSV."
        )

    df["Sentiment_clean"] = df["Sentiment"].astype(str).str.strip()
    df["Sentiment_3class"] = df["Sentiment_clean"].apply(map_to_three_classes)
    df = df[df["Sentiment_3class"].notna()].copy()

    df["clean_text"] = df["Text"].astype(str).apply(clean_text)
    return df


@st.cache_data
def load_socialmedia_data(csv_path: str) -> pd.DataFrame:
    """
    Load socialmedia.csv with columns:
    User ID, Username, Platform, Post ID, Post Text (expected).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["User ID", "Username", "Platform", "Post ID", "Post Text"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"socialmedia.csv is missing required columns: {missing}. "
            "Expected columns: " + ", ".join(required_cols)
        )

    return df


def train_test_split_data(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "Sentiment_3class",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X = df[text_col]
    y = df[label_col]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# =========================================================
# 3. Model builders
# =========================================================


def build_logreg_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), stop_words="english"
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_nb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), stop_words="english"
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str = "Model",
):
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    return {
        "model": model,
        "cv_scores": cv_scores,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }


# =========================================================
# 4. Streamlit UI
# =========================================================


def main():
    st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")

    st.title("📊 Social Media Sentiment Analysis Dashboard")
    st.markdown(
        """
This app implements the pipeline from your project plan:

- Load labeled social media data for training
- Preprocess and normalize labels into **Positive / Negative / Neutral**
- Train TF-IDF + **Logistic Regression** / **Naive Bayes**
- Evaluate performance
- Run **live predictions** and **batch sentiment scoring** on `socialmedia.csv`
        """
    )

    # ------------- Sidebar controls -------------
    st.sidebar.header("⚙️ Settings")

    csv_path = st.sidebar.text_input(
        "Training CSV path",
        value="sentimentdataset.csv",
        help="Labeled dataset. Must contain 'Text' and 'Sentiment' columns.",
    )

    socialmedia_path = st.sidebar.text_input(
        "socialmedia.csv path",
        value="socialmedia.csv",
        help="Unlabeled social media posts. Needs columns: User ID, Username, Platform, Post ID, Post Text.",
    )

    test_size = st.sidebar.slider(
        "Test size (fraction for test set)",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
    )

    models_to_train = st.sidebar.multiselect(
        "Models to train",
        options=["Logistic Regression", "Naive Bayes"],
        default=["Logistic Regression", "Naive Bayes"],
    )

    show_raw_data = st.sidebar.checkbox("Show raw training data preview", value=True)

    # For session state (trained models & data)
    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}

    # ------------- Section 1: Load training data -------------
    st.header("1. Training Data Loading & Exploration")

    data_load_col1, data_load_col2 = st.columns([2, 1])

    with data_load_col1:
        if st.button("Load & Prepare Training Data"):
            try:
                df = load_and_prepare_data(csv_path)
                st.session_state["df"] = df
                st.success(f"Loaded {len(df)} rows from `{csv_path}`.")
            except Exception as e:
                st.error(f"Error loading training data: {e}")

    with data_load_col2:
        if "df" in st.session_state:
            df = st.session_state["df"]
            st.metric("Rows", len(df))
            st.metric("Distinct labels (3-class)", df["Sentiment_3class"].nunique())

    if "df" in st.session_state:
        df = st.session_state["df"]

        if show_raw_data:
            st.subheader("Sample of Prepared Training Data")
            st.dataframe(
                df[["Text", "Sentiment_clean", "Sentiment_3class", "clean_text"]].head(
                    20
                )
            )

        # Class distribution
        st.subheader("Class Distribution (3-class labels)")
        class_counts = df["Sentiment_3class"].value_counts()
        st.bar_chart(class_counts)

        # Text length distribution
        st.subheader("Text Length Distribution")
        df["text_len"] = df["clean_text"].str.split().apply(len)
        st.caption("Number of tokens per post (after cleaning).")
        st.line_chart(df["text_len"])

    else:
        st.info("Load training data to continue.")

    # ------------- Section 2: Model training -------------
    st.header("2. Train Models")

    if "df" in st.session_state:
        df = st.session_state["df"]

        col_train_1, col_train_2 = st.columns([2, 1])
        with col_train_1:
            if st.button("Train selected models"):
                X_train, X_test, y_train, y_test = train_test_split_data(
                    df,
                    test_size=test_size,
                )
                st.session_state["split"] = (X_train, X_test, y_train, y_test)

                trained_models = {}

                if "Logistic Regression" in models_to_train:
                    with st.spinner("Training Logistic Regression..."):
                        logreg_pipe = build_logreg_pipeline()
                        result_lr = evaluate_model(
                            logreg_pipe,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            model_name="Logistic Regression",
                        )
                        trained_models["Logistic Regression"] = result_lr

                if "Naive Bayes" in models_to_train:
                    with st.spinner("Training Naive Bayes..."):
                        nb_pipe = build_nb_pipeline()
                        result_nb = evaluate_model(
                            nb_pipe,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            model_name="Naive Bayes",
                        )
                        trained_models["Naive Bayes"] = result_nb

                st.session_state["trained_models"] = trained_models
                st.success("Training completed.")

        with col_train_2:
            if "split" in st.session_state:
                X_train, X_test, y_train, y_test = st.session_state["split"]
                st.metric("Train size", len(X_train))
                st.metric("Test size", len(X_test))

        # Show metrics
        if st.session_state["trained_models"]:
            st.subheader("Model Performance")

            for name, res in st.session_state["trained_models"].items():
                st.markdown(f"### {name}")
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.metric("Test Accuracy", f"{res['accuracy']:.3f}")
                    st.metric("CV Mean Accuracy", f"{res['cv_scores'].mean():.3f}")
                    st.caption(f"CV scores: {np.round(res['cv_scores'], 3)}")

                with col2:
                    st.text("Classification Report")
                    st.text(res["report"])

                # Confusion matrix plot
                cm = res["confusion_matrix"]
                labels = res["labels"]

                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation="nearest")
                ax.set_title(f"{name} - Confusion Matrix")
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                ax.set_ylabel("True label")
                ax.set_xlabel("Predicted label")

                # text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha="center", va="center")

                st.pyplot(fig)

    # ------------- Section 3: Live & Batch Sentiment -------------
    st.header("3. Live & Batch Sentiment Prediction")

    left_col, right_col = st.columns(2)

    # ---- Left: Live prediction on user input ----
    with left_col:
        st.subheader("3.1 Live Sentiment Prediction (manual input)")

        if st.session_state["trained_models"]:
            model_name_for_inference = st.selectbox(
                "Choose model for prediction",
                options=list(st.session_state["trained_models"].keys()),
                key="live_model_select",
            )

            user_text = st.text_area(
                "Enter a social media post / text:",
                "",
                height=120,
                key="live_text_input",
            )

            if st.button("Predict sentiment", key="live_predict_button"):
                if not user_text.strip():
                    st.warning("Please enter some text.")
                else:
                    model = st.session_state["trained_models"][
                        model_name_for_inference
                    ]["model"]
                    cleaned = clean_text(user_text)
                    pred = model.predict([cleaned])[0]
                    st.success(f"Predicted sentiment: **{pred}**")
                    st.caption(
                        f"Model: {model_name_for_inference} | Cleaned text: `{cleaned}`"
                    )
        else:
            st.info("Train at least one model in Section 2 to enable live prediction.")

    # ---- Right: socialmedia.csv preview + batch scoring ----
    with right_col:
        st.subheader("3.2 socialmedia.csv Preview & Batch Scoring")

        # Load socialmedia.csv
        if st.button("Load socialmedia.csv"):
            try:
                sm_df = load_socialmedia_data(socialmedia_path)
                st.session_state["sm_df"] = sm_df
                st.success(f"Loaded {len(sm_df)} rows from `{socialmedia_path}`.")
            except Exception as e:
                st.error(f"Error loading socialmedia.csv: {e}")

        if "sm_df" in st.session_state:
            sm_df = st.session_state["sm_df"]

            st.markdown("**Preview of socialmedia.csv**")
            st.dataframe(sm_df.head(20))

            if st.session_state["trained_models"]:
                batch_model_name = st.selectbox(
                    "Choose model for batch sentiment scoring",
                    options=list(st.session_state["trained_models"].keys()),
                    key="batch_model_select",
                )

                if st.button("Run sentiment on all Post Text and generate file"):
                    model = st.session_state["trained_models"][batch_model_name][
                        "model"
                    ]

                    # Clean Post Text and predict
                    sm_df_clean = sm_df.copy()
                    sm_df_clean["clean_text"] = (
                        sm_df_clean["Post Text"].astype(str).apply(clean_text)
                    )
                    preds = model.predict(sm_df_clean["clean_text"])

                    # Prepare result DataFrame with requested columns
                    result_cols = [
                        "User ID",
                        "Username",
                        "Platform",
                        "Post ID",
                        "Post Text",
                    ]
                    missing = [c for c in result_cols if c not in sm_df_clean.columns]
                    if missing:
                        st.error(
                            f"Cannot create result file. Missing columns in socialmedia.csv: {missing}"
                        )
                    else:
                        result_df = sm_df_clean[result_cols].copy()
                        result_df["Sentiment"] = preds

                        st.success("Sentiment scoring completed.")
                        st.markdown("**Preview of scored data:**")
                        st.dataframe(result_df.head(20))

                        # Create downloadable CSV
                        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download scored socialmedia CSV",
                            data=csv_bytes,
                            file_name="socialmedia_scored_with_sentiment.csv",
                            mime="text/csv",
                        )
            else:
                st.info(
                    "Train at least one model in Section 2 to enable batch scoring."
                )
        else:
            st.info(
                "Click **Load socialmedia.csv** to preview and enable batch scoring."
            )


if __name__ == "__main__":
    main()
