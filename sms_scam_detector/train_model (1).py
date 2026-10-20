"""
SMS Scam Detection - fine-tuned training pipeline (v2)
------------------------------------------------------
What changed compared with v1:
  1. Wider hyper-parameter search (C, class weight, min_df, vocabulary size)
  2. Decision-threshold tuning on out-of-fold predictions, so the cut-off
     between "scam" and "safe" is chosen deliberately instead of fixed at 0.5.
     It favours high precision (few safe messages wrongly flagged).
  3. Extra modern scam/safe examples (mobile money, bank, delivery, OTP,
     crypto) added to TRAINING data only. The test set stays 100% original,
     so the reported accuracy is still honest.
  4. Fixed random seeds, saved report (training_report.json), cleaner code.
  5. Threshold is exported to model.js; predictor.js reads it.

Usage:
    pip install scikit-learn pandas
    python train_model.py
    python train_model.py --data spam.csv --min-precision 0.98
    python train_model.py --extra my_messages.csv     # your own labeled data

--extra file format: CSV with columns label,message (label = spam or ham).
"""
import argparse
import json
import re
import urllib.request
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")

STOPWORDS = set(
    ("a an the and or but if is are was were be been am i me my you your we our "
     "they them he she it its to of in on at for with as by from this that these "
     "those so do does did have has had will would can could should just not no yes"
     ).split()
)

# ----------------------------------------------------------------------
# Modern examples (training only). Add your own region's scams here or
# pass them with --extra. More real examples = better accuracy.
# ----------------------------------------------------------------------
MODERN_SPAM = [
    "Dear customer your GTBank account has been restricted. Verify your BVN and PIN at http://gtb-verify.online now",
    "M-Pesa: you have received Ksh 5,000 by mistake. Please reverse it immediately to 0722000000",
    "MTN promo: you have won 1,000,000 naira. Send your name and bank details to claim your prize",
    "Airtel: your line will be blocked in 24 hours. Click http://airtel-kyc.net to update your KYC now",
    "DHL: your parcel is held at customs. Pay the clearance fee of $3.50 at http://dhl-track.info to release it",
    "Your OTP request was approved. Send the 6 digit code you received to confirm your refund",
    "Bank alert: a transfer of 250,000 is pending. If this was not you, log in at http://secure-banking.xyz",
    "Congratulations! You have been shortlisted for a loan of 500,000. No collateral. Pay 5,000 processing fee to receive it",
    "Your WhatsApp account will be suspended. Verify now at http://whatsapp-verify.co to keep your account",
    "Double your bitcoin in 24 hours guaranteed. Send 0.1 BTC to the address below and receive 0.2 BTC",
    "Amazon: your order of $899 was placed. If you did not authorize this call 0800 123 456 to cancel",
    "Dear user your electricity token has expired. Pay the outstanding bill at http://power-pay.biz to avoid disconnection",
    "Work from home and earn $200 per day. Just like and share videos. Message us on Telegram to start",
    "FINAL WARNING: your tax refund of $1,240 is ready. Confirm your bank details at http://refund-gov.com",
    "Your ATM card has been deactivated. Call 08012345678 with your card number and PIN to reactivate it",
    "Hello dear, I am a widow with 5 million dollars. I need a trustworthy person to help me transfer it",
    "You are the winner of the Google anniversary draw. Send your full name and account number to collect your 100,000",
    "Netflix: your subscription is on hold. Update payment at http://netflix-help.xyz within 24 hours",
    "Congratulations, you are eligible for a free government grant. Pay a small registration fee to receive it",
    "Your SIM card registration is incomplete. Send your NIN and PIN to this number or the line will be barred",
    "Hi, I accidentally sent you money. Please send it back to this number or I will report you to the police",
    "Crypto giveaway! Send 1 ETH and get 3 ETH back instantly. Limited offer for the first 100 people",
    "Your Facebook page violates our terms. Appeal here http://fb-appeal.net or your page will be deleted today",
    "Student loan forgiveness approved. Call now to claim 10000 dollars. Limited time only",
    "URGENT: Your account has been compromised. Click http://bank-secure-login.xyz and enter your password now",
    "Get paid to test products at home. Pay 2,000 registration to receive your first payment today",
    "Claim your free data bundle 20GB now. Click http://free-data-gift.biz and enter your pin",
    "Package delivery failed. Reschedule at http://post-redeliver.info and pay a 1.99 fee",
    "Your Apple ID has been locked. Verify your identity immediately at http://apple-id-unlock.com",
    "Dear beneficiary, your inheritance fund of 4.5 million is ready for release. Contact our lawyer with your details",
]

MODERN_HAM = [
    "Your OTP is 482913. It expires in 10 minutes. Do not share this code with anyone",
    "Your verification code is 771204. If you did not request this, ignore this message",
    "Your order has shipped and should arrive on Thursday. Thanks for shopping with us",
    "Reminder: your dentist appointment is tomorrow at 9:30am. Reply C to confirm",
    "Hi, this is a reminder that your electricity bill is due on the 25th. Thank you",
    "Your uber is arriving in 3 minutes. Driver is in a white Toyota Corolla",
    "Hey, I just sent the money for the groceries. Check your account when you can",
    "Hi Sarah, your prescription is ready for pickup at the pharmacy until Friday",
    "Meeting moved to 3pm in room 204. Please bring the printed slides",
    "Can you transfer me 5000 for the taxi? I'll pay you back on Monday",
    "Thanks for your payment. Your receipt number is 884512. Have a great day",
    "Your flight BA074 departs at 18:40 from gate 12. Boarding starts at 18:00",
    "Dinner at my place on Saturday? I'm making jollof rice",
    "Hi, the plumber will come between 10am and 12pm tomorrow. Please be home",
    "Your package was delivered to the front door at 2:15pm. Enjoy",
    "Mum I have arrived at school. I will call you after class",
    "Don't forget the church meeting this Sunday after service",
    "Your monthly statement is now available in the app. Log in to view it",
    "We have received your application and will get back to you within 5 working days",
    "Happy new month! Wishing you peace and prosperity",
    "Hi, can you send me the invoice for last month's work? I need it for accounts",
    "Your table for 2 at 8pm is confirmed. See you tonight",
    "Rent is due on the 1st. Please send the transfer receipt when done",
    "I'm outside the gate, can you come down?",
    "Class is cancelled today because the lecturer is sick. Check the group chat for updates",
    "Your appointment with Dr. Okafor has been rescheduled to Wednesday at 11am",
    "Thanks for the birthday wishes everyone! Really appreciate it",
    "The match starts at 7. I'll save you a seat at the viewing centre",
    "Please find the meeting notes attached. Let me know if anything is missing",
    "Your internet bill payment was successful. Thank you for being our customer",
]


# ----------------------------------------------------------------------
# Preprocessing (must stay identical to tokenize() in predictor.js)
# ----------------------------------------------------------------------
def tokenize(text: str):
    t = text.lower()
    t = re.sub(r"https?://\S+|www\.\S+|\b\S+\.(?:com|biz|xyz|info|co|net)\b\S*",
               " urltoken ", t)
    t = re.sub(r"[$£€]\s?\d[\d,.]*|\d[\d,.]*\s?(?:usd|dollars|pounds)",
               " moneytoken ", t)
    t = re.sub(r"\b\d{5,}\b", " longnumtoken ", t)
    t = re.sub(r"\b\d+\b", " numtoken ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [w for w in t.split() if len(w) > 1 and w not in STOPWORDS]
    if re.search(r"[A-Z]{4,}", text):
        tokens.append("allcapstoken")
    if re.search(r"!{2,}", text):
        tokens.append("excitetoken")
    return tokens


def analyzer(text: str):
    """Unigrams + bigrams. Bigrams capture phrases like 'claim prize'."""
    toks = tokenize(text)
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def parse_table(raw: str) -> pd.DataFrame:
    first_line = raw.splitlines()[0].lower()
    if "\t" in first_line and not first_line.startswith(("v1", "label")):
        df = pd.read_csv(StringIO(raw), sep="\t", header=None,
                         names=["label", "message"])
    else:
        df = pd.read_csv(StringIO(raw))
        df = df.rename(columns={"v1": "label", "v2": "message",
                                "Category": "label", "Message": "message"})
        df = df[["label", "message"]]
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].dropna()
    return df


def load_data(path: Optional[str]) -> pd.DataFrame:
    if path:
        raw = open(path, "r", encoding="latin-1").read()
    else:
        print(f"Downloading dataset from {DATA_URL}")
        raw = urllib.request.urlopen(DATA_URL).read().decode("utf-8", "ignore")
    df = parse_table(raw).drop_duplicates(subset="message").reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype(int)
    print(f"Loaded {len(df)} unique messages "
          f"({df['y'].sum()} spam, {(1 - df['y']).sum()} ham)")
    return df


def extra_frame(extra_path: Optional[str]) -> pd.DataFrame:
    rows = [("spam", m) for m in MODERN_SPAM] + [("ham", m) for m in MODERN_HAM]
    df = pd.DataFrame(rows, columns=["label", "message"])
    if extra_path:
        user = parse_table(open(extra_path, "r", encoding="utf-8").read())
        df = pd.concat([df, user], ignore_index=True)
    df = df.drop_duplicates(subset="message").reset_index(drop=True)
    df["y"] = (df["label"] == "spam").astype(int)
    print(f"Extra training examples: {len(df)} "
          f"({df['y'].sum()} spam, {(1 - df['y']).sum()} ham)")
    return df


# ----------------------------------------------------------------------
# Threshold tuning
# ----------------------------------------------------------------------
def pick_threshold(y_true, probs, min_precision: float) -> float:
    """Highest-F1 threshold that keeps precision >= min_precision.
    Falls back to the best-F1 threshold if the precision target is unreachable."""
    best_t, best_f1, fallback_t, fallback_f1 = 0.5, -1.0, 0.5, -1.0
    for t in np.arange(0.20, 0.96, 0.01):
        pred = (probs >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred)
        if f > fallback_f1:
            fallback_t, fallback_f1 = t, f
        if p >= min_precision and f > best_f1:
            best_t, best_f1 = t, f
    return float(round(best_t if best_f1 >= 0 else fallback_t, 2))


def report(name, y_true, pred):
    cm = confusion_matrix(y_true, pred)
    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_true, pred):.4f}")
    print(f"  Precision: {precision_score(y_true, pred):.4f}  "
          f"Recall: {recall_score(y_true, pred):.4f}  "
          f"F1: {f1_score(y_true, pred):.4f}")
    print(f"  Confusion [[TN FP] [FN TP]]: {cm.tolist()}")
    return {"accuracy": round(accuracy_score(y_true, pred), 4),
            "precision": round(precision_score(y_true, pred), 4),
            "recall": round(recall_score(y_true, pred), 4),
            "f1": round(f1_score(y_true, pred), 4),
            "confusion": cm.tolist()}


def make_vectorizer(min_df=2, max_features=8000):
    return TfidfVectorizer(analyzer=analyzer, sublinear_tf=True,
                           min_df=min_df, max_features=max_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to local dataset (optional)")
    ap.add_argument("--extra", help="CSV (label,message) of your own labeled messages")
    ap.add_argument("--out", default="model.js", help="output file for the web app")
    ap.add_argument("--extra-weight", type=int, default=6,
                    help="how many times the extra examples are repeated in training")
    ap.add_argument("--min-precision", type=float, default=0.97,
                    help="precision floor used when choosing the threshold")
    args = ap.parse_args()

    df = load_data(args.data)
    extra = extra_frame(args.extra)

    # Test set comes ONLY from the original dataset so it stays honest.
    X_tr, X_test, y_tr, y_test = train_test_split(
        df["message"], df["y"], test_size=0.2, stratify=df["y"], random_state=SEED)
    # Extra examples are repeated so a few dozen modern messages are not
    # drowned out by ~4,000 older ones (see --extra-weight).
    extra_rep = pd.concat([extra] * args.extra_weight, ignore_index=True)
    X_train = pd.concat([X_tr, extra_rep["message"]], ignore_index=True)
    y_train = pd.concat([y_tr, extra_rep["y"]], ignore_index=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- 1. compare model families ----
    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, class_weight="balanced",
                                                  max_iter=3000),
        "Linear SVM": LinearSVC(C=1, class_weight="balanced"),
    }
    print("\n5-fold cross-validation (F1 on scam class):")
    for name, clf in candidates.items():
        pipe = Pipeline([("tfidf", make_vectorizer()), ("clf", clf)])
        s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        print(f"  {name:<20} F1 = {s.mean():.4f} (+/- {s.std():.4f})")

    # ---- 2. tune Logistic Regression (exportable to JavaScript) ----
    pipe = Pipeline([("tfidf", make_vectorizer()),
                     ("clf", LogisticRegression(max_iter=5000))])
    grid = GridSearchCV(
        pipe,
        param_grid={
            "clf__C": [3, 10, 30, 100, 300],
            "clf__class_weight": [None, "balanced", {0: 1, 1: 3}],
            "tfidf__min_df": [1, 2],
            "tfidf__max_features": [8000, 15000],
        },
        cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_params_
    print(f"\nBest parameters: {best}")
    print(f"Best CV F1:      {grid.best_score_:.4f}")

    # ---- 3. choose the decision threshold on out-of-fold probabilities ----
    oof = cross_val_predict(grid.best_estimator_, X_train, y_train, cv=cv,
                            method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train.values, oof, args.min_precision)
    print(f"Chosen threshold: {threshold} (precision floor {args.min_precision})")

    # ---- 4. honest evaluation on the untouched test set ----
    probs = grid.predict_proba(X_test)[:, 1]
    res_default = report("Test set @ threshold 0.50", y_test, (probs >= 0.5).astype(int))
    res_tuned = report(f"Test set @ tuned threshold {threshold}", y_test,
                       (probs >= threshold).astype(int))
    print("\n" + classification_report(
        y_test, (probs >= threshold).astype(int), target_names=["ham", "spam"]))

    # ---- 5. spot-check modern messages the old data never covered ----
    probe = [
        "Dear customer your account is restricted. Verify your BVN at http://bank-check.online",
        "M-Pesa: you received Ksh 2,000 by mistake, please send it back now",
        "Hey are you coming for the meeting later?",
        "Your OTP is 553821. Do not share it with anyone",
    ]
    pp = grid.predict_proba(pd.Series(probe))[:, 1]
    print("Spot check (scam probability):")
    for m, p in zip(probe, pp):
        print(f"  {p:.3f}  {'SCAM' if p >= threshold else 'safe'}  | {m[:60]}")

    # ---- 6. retrain on ALL data (original + extra) and export ----
    all_X = pd.concat([df["message"], extra_rep["message"]], ignore_index=True)
    all_y = pd.concat([df["y"], extra_rep["y"]], ignore_index=True)
    vec = make_vectorizer(best["tfidf__min_df"], best["tfidf__max_features"])
    Xall = vec.fit_transform(all_X)
    clf = LogisticRegression(C=best["clf__C"], class_weight=best["clf__class_weight"],
                             max_iter=5000).fit(Xall, all_y)

    terms = vec.get_feature_names_out()
    model = {
        "intercept": round(float(clf.intercept_[0]), 6),
        "threshold": threshold,
        "terms": {t: [round(float(vec.idf_[i]), 5), round(float(clf.coef_[0][i]), 5)]
                  for i, t in enumerate(terms)},
        "meta": {
            "trainedOn": int(len(all_X)),
            "testAccuracy": res_tuned["accuracy"],
            "testF1": res_tuned["f1"],
            "params": {k: str(v) for k, v in best.items()},
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("window.TRAINED_MODEL = " + json.dumps(model, separators=(",", ":")) + ";\n")
    with open("training_report.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": {k: str(v) for k, v in best.items()},
                   "cv_f1": round(float(grid.best_score_), 4),
                   "threshold": threshold,
                   "test_at_0.5": res_default,
                   "test_at_threshold": res_tuned}, f, indent=2)
    print(f"\nSaved {args.out} ({len(terms)} features) and training_report.json")
    order = clf.coef_[0].argsort()[::-1][:15]
    print("Top scam indicators:", ", ".join(terms[i] for i in order))


if __name__ == "__main__":
    main()
