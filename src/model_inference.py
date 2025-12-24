# src/model_inference.py
import joblib


def load_svm_pipeline(model_path="model/SVM.joblib"):
    """
    加载 sklearn Pipeline（tfidf + LinearSVC）
    """
    pipe = joblib.load(model_path)

    # 可选：简单校验
    if not hasattr(pipe, "predict"):
        raise ValueError("Loaded object is not a sklearn Pipeline")

    return pipe


def svm_predict(texts, pipe):
    """
    Parameters
    ----------
    texts : List[str]
        英文评论列表
    pipe : sklearn Pipeline
        tfidf + LinearSVC

    Returns
    -------
    List[dict]
        [{text, label, score}]
    """
    scores = pipe.decision_function(texts)
    preds = pipe.predict(texts)

    results = []
    for text, y, s in zip(texts, preds, scores):
        label = "deceptive" if y == 1 else "truthful"
        results.append({
            "text": text,
            "label": label,
            "score": float(s)
        })

    return results
