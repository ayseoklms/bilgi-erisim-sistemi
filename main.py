from utils import read_documents, tokenize_documents, build_vocabulary, vectorize_documents
from inverted_index import build_inverted_index
from search import and_query, or_query, not_query, preprocess_query
from tfidf import compute_tf_idf, compute_cosine_similarity
from evaluation import evaluate
from colorama import init, Fore, Style
import csv

init(autoreset=True)

# === 1. Eğitim verisini oku ve işle ===
train_folder = "data/train"
documents, doc_id_map = read_documents(train_folder)
tokenized_docs = tokenize_documents(documents)
vocab = build_vocabulary(tokenized_docs)
vectors, labels = vectorize_documents(tokenized_docs, vocab)
inverted_index = build_inverted_index(tokenized_docs)
tf_idf_vectors, idf = compute_tf_idf(tokenized_docs)

# === 2. Kullanıcıdan sorgu al ===
query = input(Fore.YELLOW + "Sorgunuzu girin: ")
query_tokens = preprocess_query(query)
query_type = input(Fore.YELLOW + "Sorgu tipi (AND / OR / NOT): ").strip().upper()

# === 3. Boolean sorguyu uygula ===
if query_type == "AND":
    results = and_query(query_tokens, inverted_index)
elif query_type == "OR":
    results = or_query(query_tokens, inverted_index)
elif query_type == "NOT":
    if len(query_tokens) != 1:
        print(Fore.RED + "NOT sorgusu yalnızca tek bir terim içermelidir.")
        results = []
    else:
        results = not_query(query_tokens[0], len(documents), inverted_index)
else:
    print(Fore.RED + "Geçersiz sorgu tipi.")
    results = []

print(Fore.CYAN + f"\n{query_type} Query Results (Doc IDs): {results}")

# === 4. Cosine similarity ile en yakın belgeleri sırala ===
print(Fore.MAGENTA + "\nBenzerlik Sıralaması (TF-IDF Cosine Similarity):")
query_vector = {term: idf.get(term, 0) for term in query_tokens}
similarities = []
for doc_id, doc_vector in enumerate(tf_idf_vectors):
    sim = compute_cosine_similarity(query_vector, doc_vector)
    similarities.append((doc_id, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
top_results = similarities[:10]

for doc_id, sim in top_results:
    print(Fore.GREEN + f"Doc ID: {doc_id}, Dosya: {doc_id_map[doc_id]}, Benzerlik: {sim:.4f}")

# === 5. TEST VERİSİ ÜZERİNDE DEĞERLENDİRME ===
print(Fore.BLUE + "\n--- TEST VERİSİ DEĞERLENDİRME ---")
test_folder = "data/test"
test_docs, test_doc_id_map = read_documents(test_folder)
tokenized_test = tokenize_documents(test_docs)
test_vectors, test_labels = vectorize_documents(tokenized_test, vocab)

predicted_labels = []
true_labels = []

for i, (tokens, true_label) in enumerate(tokenized_test):
    test_query = preprocess_query(query)
    query_vec = {term: idf.get(term, 0) for term in test_query}

    sims = []
    for doc_id, doc_vector in enumerate(tf_idf_vectors):
        sim = compute_cosine_similarity(query_vec, doc_vector)
        sims.append((doc_id, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_doc_id = sims[0][0]
    predicted_labels.append(labels[top_doc_id])
    true_labels.append(true_label)

metrics = evaluate(true_labels, predicted_labels)
print(Fore.GREEN + f"Precision: {metrics['precision']:.4f}")
print(Fore.GREEN + f"Recall: {metrics['recall']:.4f}")
print(Fore.GREEN + f"F1 Score: {metrics['f1']:.4f}")

# === 6. SONUÇLARI CSV DOSYASINA YAZ ===
with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Precision", "Recall", "F1 Score"])
    writer.writerow([metrics['precision'], metrics['recall'], metrics['f1']])
print(Fore.YELLOW + "\nSonuçlar 'results.csv' dosyasına kaydedildi.")
