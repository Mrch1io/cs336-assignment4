import hashlib
import os
import re
import unicodedata
from collections import Counter, defaultdict

import mmh3


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    counts = Counter()
    for path in input_files:
        with open(path) as f:
            for line in f:
                counts[hashlib.md5(line.encode()).hexdigest()] += 1
    for path in input_files:
        with open(path) as f_in, open(os.path.join(output_directory, os.path.basename(path)), "w") as f_out:
            for line in f_in:
                if counts[hashlib.md5(line.encode()).hexdigest()] == 1:
                    f_out.write(line)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def get_ngrams(text: str, n: int) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    ngram_sets = []
    signatures = []
    for path in input_files:
        with open(path) as f:
            ngram_set = get_ngrams(f.read(), ngrams)
        ngram_sets.append(ngram_set)
        signatures.append([min(mmh3.hash(ngram, seed) for ngram in ngram_set) for seed in range(num_hashes)])

    rows = num_hashes // num_bands
    buckets = defaultdict(list)
    for i, signature in enumerate(signatures):
        for band in range(num_bands):
            buckets[(band, tuple(signature[band * rows:(band + 1) * rows]))].append(i)

    parent = list(range(len(input_files)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for docs in buckets.values():
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                a, b = ngram_sets[docs[i]], ngram_sets[docs[j]]
                if len(a & b) / len(a | b) >= jaccard_threshold:
                    parent[find(docs[j])] = find(docs[i])

    for i, path in enumerate(input_files):
        if find(i) == i:
            with open(path) as f_in, open(os.path.join(output_directory, os.path.basename(path)), "w") as f_out:
                f_out.write(f_in.read())
