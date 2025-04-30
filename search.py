#Boolean retrieval

# search.py

from utils import preprocess_tokens

def and_query(terms, inverted_index):
    if not terms:
        return []
    result = set(doc_id for doc_id, _ in inverted_index.get(terms[0], []))
    for term in terms[1:]:
        result &= set(doc_id for doc_id, _ in inverted_index.get(term, []))
    return sorted(result)

def or_query(terms, inverted_index):
    result = set()
    for term in terms:
        result |= set(doc_id for doc_id, _ in inverted_index.get(term, []))
    return sorted(result)

def not_query(term, total_docs, inverted_index):
    all_docs = set(range(total_docs))
    term_docs = set(doc_id for doc_id, _ in inverted_index.get(term, []))
    return sorted(all_docs - term_docs)


def preprocess_query(query):
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(query.lower())
    return preprocess_tokens(tokens)

