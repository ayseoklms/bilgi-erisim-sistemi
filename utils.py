#Veri okuma ve ön işleme klasörü


# utils.py

### === utils.py ===

import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

# Gerekli NLTK kaynaklarını indir
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def read_documents(folder_path, max_docs=1000):
    documents = []
    doc_id_map = {}
    count = 0
    for label in ['pos', 'neg']:
        label_path = os.path.join(folder_path, label)
        for filename in os.listdir(label_path):
            if count >= max_docs:
                return documents, doc_id_map
            file_path = os.path.join(label_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append((content, label))
                doc_id_map[count] = filename
                count += 1
    return documents, doc_id_map

def tokenize_documents(documents):
    tokenized_docs = []
    for content, label in documents:
        tokens = word_tokenize(content.lower())
        tokenized_docs.append((tokens, label))
    return tokenized_docs

def preprocess_tokens(tokens):
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def build_vocabulary(tokenized_docs):
    vocab = set()
    for tokens, _ in tokenized_docs:
        processed = preprocess_tokens(tokens)
        vocab.update(processed)
    return sorted(list(vocab))

def vectorize_documents(tokenized_docs, vocab):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    vectors = []
    labels = []
    for tokens, label in tokenized_docs:
        processed = preprocess_tokens(tokens)
        vector = [0] * len(vocab)
        for token in processed:
            if token in vocab_index:
                vector[vocab_index[token]] += 1
        vectors.append(vector)
        labels.append(label)
    return vectors, labels


