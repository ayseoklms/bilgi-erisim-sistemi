def evaluate(true_labels, predicted_labels):
    tp = sum(t == p == 'pos' for t, p in zip(true_labels, predicted_labels))
    tn = sum(t == p == 'neg' for t, p in zip(true_labels, predicted_labels))
    fp = sum(t == 'neg' and p == 'pos' for t, p in zip(true_labels, predicted_labels))
    fn = sum(t == 'pos' and p == 'neg' for t, p in zip(true_labels, predicted_labels))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}