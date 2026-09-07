import numpy as np

def calculate_edit_distance(r, h):
    """
    Calculates the Levenshtein distance between reference sequence r and hypothesis sequence h.
    r, h: lists of tokens
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint32)
    d = d.reshape((len(r) + 1, len(h) + 1))
    
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
        
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
                
    return d[len(r)][len(h)]

def calculate_wer(reference, hypothesis):
    """
    Word Error Rate calculation.
    reference: list of lists of words (or lists of tokens)
    hypothesis: list of lists of words (or lists of tokens)
    """
    total_distance = 0
    total_words = 0
    
    for r, h in zip(reference, hypothesis):
        total_distance += calculate_edit_distance(r, h)
        total_words += len(r)
        
    if total_words == 0:
        return 0.0 if total_distance == 0 else 1.0
        
    return float(total_distance) / total_words

def get_ngrams(sequence, n):
    ngrams = {}
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

def calculate_bleu(references, hypotheses, max_n=4):
    """
    Calculates BLEU score for a list of hypotheses and references.
    references: list of lists of strings (single reference per hypothesis)
    hypotheses: list of lists of strings
    """
    p_ns = []
    for n in range(1, max_n + 1):
        total_matches = 0
        total_hypothesis_ngrams = 0
        
        for r, h in zip(references, hypotheses):
            h_ngrams = get_ngrams(h, n)
            r_ngrams = get_ngrams(r, n)
            
            matches = 0
            for ngram, count in h_ngrams.items():
                if ngram in r_ngrams:
                    matches += min(count, r_ngrams[ngram])
            
            total_matches += matches
            total_hypothesis_ngrams += sum(h_ngrams.values())
            
        if total_hypothesis_ngrams == 0:
            p_n = 0.0
        else:
            p_n = total_matches / total_hypothesis_ngrams
        p_ns.append(p_n)
        
    # Calculate Brevity Penalty (BP)
    r_len = sum(len(r) for r in references)
    h_len = sum(len(h) for h in hypotheses)
    
    if h_len == 0:
        return 0.0
        
    if h_len > r_len:
        bp = 1.0
    else:
        bp = np.exp(1 - r_len / h_len)
        
    # Geometric mean of p_ns
    if any(p == 0 for p in p_ns):
        return 0.0
        
    geometric_mean = np.exp(np.mean([np.log(p) for p in p_ns]))
    return bp * geometric_mean
