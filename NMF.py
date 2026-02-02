import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# Path to the plain-text book
PATH = "War and Peace.txt"
text = Path(PATH).read_text(encoding="utf-8", errors="ignore")

# Split the book into chapters
chapters = re.split(r"\n\s*CHAPTER\s+[IVXLCDM]+\s*\n", text)

# Build a TF-IDF matrix. Keep the configuration compact so it runs quickly
# but still remove very rare and overly common words.
stop = set(ENGLISH_STOP_WORDS) | {"chapter", "said", "mr", "mrs"}
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=list(stop),
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # only words with >=3 letters
    min_df=3,                               # ignore words that appear in <3 documents
    max_df=0.5,                             # ignore words that appear in >60% of documents
    ngram_range=(1, 2),                     # unigrams + bigrams
    max_features=30000,                     # cap vocabulary for speed/stability (try 20000–50000)
    sublinear_tf=True,                      # tf := 1 + log(tf)
)

X = vectorizer.fit_transform(chapters)
terms = np.array(vectorizer.get_feature_names_out())
print("TF-IDF matrix:", X.shape)

# Factorize with NMF to get K topics
K = 8
nmf = NMF(
    n_components=K,
    init="nndsvda",
    random_state=42,
    max_iter=800,
)

W = nmf.fit_transform(X)     # document-topic matrix (n_docs x K)
H = nmf.components_          # topic-term matrix (K x n_terms)

# Normalize rows of W so each chapter's topic weights sum to 1
Wn = normalize(W, norm="l1", axis=1)

# Print the most-weighted words for each topic
TOP_WORDS = 12
print("\n=== TOPICS (top words) ===")
for k in range(K):
    top_idx = np.argsort(H[k])[::-1][:TOP_WORDS]
    print(f"Topic {k:02d}: " + ", ".join(terms[top_idx]))

# For each topic, show a couple of chapters where that topic is dominant.
dominant_topic = Wn.argmax(axis=1)
dominant_weight = Wn.max(axis=1)

print("\n=== REPRESENTATIVE CHAPTERS ===")
for k in range(K):
    idx = np.where(dominant_topic == k)[0]
    if len(idx) == 0:
        continue

    # Select the two chapters with the highest share of this topic
    top = idx[np.argsort(dominant_weight[idx])[::-1][:2]]

    print(f"\n-- Topic {k:02d} (best chapters) --")
    for i in top:
        snippet = chapters[i][:220].replace("\n", " ")
        print(f"chapter={i:03d} weight={dominant_weight[i]:.3f} | {snippet}...")