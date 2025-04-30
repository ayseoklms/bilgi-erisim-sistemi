#Ters indeks oluşturma

# inverted_index.py

from collections import defaultdict
from utils import preprocess_tokens

def build_inverted_index(tokenized_docs):
    inverted_index = defaultdict(list)
    for doc_id, (tokens, _) in enumerate(tokenized_docs):
        processed_tokens = preprocess_tokens(tokens)
        token_freq = defaultdict(int)
        for token in processed_tokens:
            token_freq[token] += 1
        for token, freq in token_freq.items():
            inverted_index[token].append((doc_id, freq))
    return inverted_index
