"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    text=text.lower()
    text=''.join(s if s.isalpha() else ' ' for s in text).rstrip()
    return text

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    text=text.split()
    return text

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    return [tokenize(clean_text(i)) for i in texts]

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    np.random.seed(seed)
    idx=np.random.permutation(n_samples)
    n_val=int(n_samples*val_fraction)
    n_test=int(n_samples*test_fraction)

    return np.split(idx,[n_samples-n_val-n_test,n_samples-n_test])

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    f={}
    for i in tokenized_docs:
        for token in i:
            f[token]=f.get(token,0)+1
    return f

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # TODO: Keep the top max_size most frequent words; map each to an index in [0, V).
    f={}
    words=sorted(word_counts)
    words=sorted(words, key=word_counts.get, reverse=True)
    for i in range(min(len(word_counts),max_size)):
        f[words[i]]=i
    return f

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    l=np.zeros(len(vocab))
    for i in tokens:
        if i in vocab:
            l[vocab[i]]+=1
    return l

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    rows=[tokens_to_bow(doc,vocab) for doc in tokenized_docs]
    return np.array(rows).reshape(len(tokenized_docs),len(vocab))

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    return (bow_matrix>0).sum(axis=0)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    idf=np.zeros(len(df))
    for i in range(len(df)):
        idf[i]=np.log((n_docs+1)/(df[i]+1))+1
    return idf

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    return bow_matrix*idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    df=compute_document_frequencies(bow_train)
    return compute_idf(df,bow_train.shape[0])

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    return 1/(1+np.exp(-z))

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    return sigmoid(X @ w +b)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    y_prob=np.clip(y_prob,1e-15,1-1e-15)
    bce=-np.mean(y_true*np.log(y_prob)+(1-y_true)*np.log(1-y_prob))
    return float(bce+l2_lambda*np.sum(w**2)/2)

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.
    e=y_prob-y_true
    return X.T@e/len(y_true)+l2_lambda*w, float(e.mean())

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    return np.zeros(n_features,),0.0

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    y_prob=logistic_predict_proba(X,w,b)
    loss=binary_cross_entropy(y,y_prob,w,l2_lambda)
    dw,db=logistic_gradients(X,y,y_prob,w,l2_lambda)
    return w-lr*dw,b-lr*db,loss

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss...
    w,b=initialize_logistic_params(X.shape[1])
    losses=[]
    for _ in range(n_epochs):
        w,b,loss=gradient_descent_step(X,y,w,b,lr,l2_lambda)
        losses.append(loss)
    return w,b,losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    return np.asarray([1 if p>=threshold else 0 for p in proba])

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # TODO: Return the four confusion-matrix counts (tp, fp, tn, fn) as Python ints
    c=np.bincount(2*y_true+y_pred,minlength=4)
    return int(c[3]),int(c[1]),int(c[0]),int(c[2])

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    # TODO: Derive precision, recall, F1, and accuracy from confusion counts...
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    accuracy=(tp+tn)/(tp+fp+tn+fn) if tp+fp+tn+fn else 0.0
    return {'precision': float(precision),'recall': float(recall),'f1': float(f1),'accuracy': float(accuracy)}

# Step 23 - tune_decision_threshold
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    # TODO: Find the decision threshold that maximizes F1 on validation data.
    thresholds=np.linspace(0.0,1.0,101) if thresholds is None else thresholds
    scores=[metrics_from_counts(*confusion_counts(y_true, predict_labels(proba, t)))['f1'] for t in thresholds]
    i = np.argmax(scores)
    return float(thresholds[i]), float(scores[i])

# Step 24 - evaluate_predictions
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # TODO: Bundle confusion counts and classification metrics into one report dict
    tp,fp,tn,fn=confusion_counts(y_true,y_pred)
    metrics=metrics_from_counts(tp,fp,tn,fn)

    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,**metrics}

# Step 25 - vectorize_texts
def vectorize_texts(texts: list, vocab: dict, idf: np.ndarray) -> np.ndarray:
    # TODO: Clean, tokenize, BoW, and TF-IDF transform a list of raw strings.
    tokenized=tokenize_corpus(texts)
    bow=corpus_to_bow_matrix(tokenized,vocab)
    return transform_tfidf(bow,idf)

# Step 26 - predict_text
def predict_text(text: str, vocab: dict, idf: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> int:
    """Label a single raw message with the fitted classifier.

    Args:
        text: Raw input string.
        vocab: Fitted word -> column index map.
        idf: Fitted IDF vector, shape (V,).
        w: Logistic weight vector, shape (V,).
        b: Logistic bias scalar.
        threshold: Decision threshold for the positive class.

    Returns:
        Predicted label as int 0 or 1.
    """
    # TODO: label a single unseen raw message using fitted model artifacts
    features=vectorize_texts([text],vocab,idf)
    proba=logistic_predict_proba(features,w,b)
    return int(predict_labels(proba,threshold)[0])

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

