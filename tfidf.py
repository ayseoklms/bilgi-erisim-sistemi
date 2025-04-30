# TF-IDF hesaplama

import math
from collections import defaultdict
from utils import preprocess_tokens

def compute_tf_idf(tokenized_docs):
    D = len(tokenized_docs)
    df = defaultdict(int)
    tf = [defaultdict(int) for _ in range(D)]

    for doc_id, (tokens, _) in enumerate(tokenized_docs):
        processed = preprocess_tokens(tokens)
        for token in processed:
            tf[doc_id][token] += 1
        for token in set(processed):
            df[token] += 1

    idf = {term: math.log(D / df[term]) for term in df}

    tf_idf_vectors = []
    for doc_tf in tf:
        vector = {term: freq * idf[term] for term, freq in doc_tf.items()}
        tf_idf_vectors.append(vector)

    return tf_idf_vectors, idf


def compute_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    num = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denom = math.sqrt(sum1) * math.sqrt(sum2)
    if not denom:
        return 0.0
    return num / denom
