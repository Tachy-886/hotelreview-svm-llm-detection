import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_save_model(
    csv_path: str,
    model_save_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a Linear SVM deceptive-review classifier,
    evaluate on a held-out test set, then retrain on
    full data and save the final model.

    Parameters
    ----------
    csv_path : str
        Path to input CSV file (must contain 'text' and 'deceptive').
    model_save_path : str
        Path to save trained model (joblib).
    test_size : float
        Test set ratio (default 0.2).
    random_state : int
        Random seed.
    """

    # =========================
    # 1) Load data
    # =========================
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "deceptive"]).copy()

    X = df["text"].astype(str).values
    y = (df["deceptive"].str.lower() == "deceptive").astype(int).values
    # 1 = deceptive, 0 = truthful

    # =========================
    # 2) Stratified Train / Test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # =========================
    # 3) Pipeline: TF-IDF → Linear SVM
    # =========================
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    # =========================
    # 4) Train on training set
    # =========================
    pipe.fit(X_train, y_train)

    # =========================
    # 5) Evaluate on test set
    # =========================
    y_test_pred = pipe.predict(X_test)

    print("===== Test Set Evaluation =====")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=["truthful", "deceptive"]
    ))

    # =========================
    # 6) Retrain on full dataset
    # =========================
    pipe.fit(X, y)

    # =========================
    # 7) Save model
    # =========================
    joblib.dump(pipe, model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    return pipe
