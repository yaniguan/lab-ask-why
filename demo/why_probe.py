"""Ask 'why do you think so?' instead of judging who is right.

When a colleague makes a claim, a useful reply is to ask for the reasons and
look at the sentences that support or contradict it. This script tests a tiny
version of that idea on SciFact (Wadden et al., 2020): for every dev claim with
evidence, it ranks the sentences of each gold evidence abstract with BM25
(plain python/numpy) and checks whether a gold rationale sentence appears in
the top 1 or top 3. A random sentence ranking (20 seeds) is the baseline.
It also prints two examples with three template 'why' questions per claim.

Data: SciFact is released under CC BY-NC 2.0. It is downloaded to data/ and
should not be committed.

Run:  python demo/why_probe.py
"""
import json
import math
import os
import re
import tarfile
import urllib.request
from collections import Counter

import numpy as np

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA = os.path.join(ROOT, "data")
STOP = set("a an the of in on to for and or is are was were be by with as at from that this "
           "it its which than into not no".split())


def ensure_data():
    if os.path.exists(os.path.join(DATA, "claims_dev.jsonl")):
        return
    os.makedirs(ROOT, exist_ok=True)
    archive = os.path.join(ROOT, "data.tar.gz")
    if not os.path.exists(archive):
        print("Downloading SciFact ...")
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(ROOT)


def read_jsonl(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(line) for line in f]


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25 with IDF and average length taken from all corpus sentences."""

    def __init__(self, sentences, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        df = Counter()
        lengths = []
        for s in sentences:
            toks = tokenize(s)
            lengths.append(len(toks))
            df.update(set(toks))
        n = len(sentences)
        self.avgdl = float(np.mean(lengths))
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query, sentences):
        q = tokenize(query)
        out = np.zeros(len(sentences))
        for i, s in enumerate(sentences):
            toks = tokenize(s)
            tf = Counter(toks)
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            for t in q:
                if t in tf:
                    out[i] += self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
        return out


def why_questions(claim):
    c = claim.rstrip(".")
    return [
        f"What evidence makes you think that {c[0].lower() + c[1:]}?",
        f"What are you assuming must be true for '{c}' to hold?",
        f"What result would change your mind about '{c}'?",
    ]


def build_pairs(claims, corpus):
    """One item per (claim, gold evidence doc): sentences, gold indices, label."""
    pairs = []
    for c in claims:
        for doc_id, sets in c["evidence"].items():
            gold = sorted({i for s in sets for i in s["sentences"]})
            pairs.append({"claim": c["claim"], "sents": corpus[int(doc_id)]["abstract"],
                          "gold": set(gold), "label": sets[0]["label"],
                          "title": corpus[int(doc_id)]["title"]})
    return pairs


def recall_at(rankings, pairs, k):
    return float(np.mean([bool(set(r[:k]) & p["gold"]) for r, p in zip(rankings, pairs)]))


def main():
    ensure_data()
    corpus = {d["doc_id"]: d for d in read_jsonl("corpus.jsonl")}
    claims = [c for c in read_jsonl("claims_dev.jsonl") if c["evidence"]]
    pairs = build_pairs(claims, corpus)
    bm25 = BM25([s for d in corpus.values() for s in d["abstract"]])

    bm25_rank = [list(np.argsort(-bm25.scores(p["claim"], p["sents"]), kind="stable"))
                 for p in pairs]
    rand = {1: [], 3: []}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ranks = [list(rng.permutation(len(p["sents"]))) for p in pairs]
        for k in rand:
            rand[k].append(recall_at(ranks, pairs, k))

    n_sents = [len(p["sents"]) for p in pairs]
    labels = Counter(p["label"] for p in pairs)
    print("SciFact dev: rationale sentence retrieval inside gold evidence abstracts")
    print(f"claims with evidence: {len(claims)}   claim-doc pairs: {len(pairs)}   "
          f"labels: {dict(labels)}")
    print(f"sentences per abstract: mean {np.mean(n_sents):.1f}, median {np.median(n_sents):.0f}")
    print()
    print(f"{'method':<22}{'recall@1':>10}{'recall@3':>10}")
    print(f"{'BM25':<22}{recall_at(bm25_rank, pairs, 1):>10.3f}{recall_at(bm25_rank, pairs, 3):>10.3f}")
    print(f"{'random (20 seeds)':<22}{np.mean(rand[1]):>10.3f}{np.mean(rand[3]):>10.3f}")
    print(f"{'  random std':<22}{np.std(rand[1]):>10.3f}{np.std(rand[3]):>10.3f}")
    for lab in ("SUPPORT", "CONTRADICT"):
        idx = [i for i, p in enumerate(pairs) if p["label"] == lab]
        sub_p = [pairs[i] for i in idx]
        sub_r = [bm25_rank[i] for i in idx]
        print(f"{'BM25 ' + lab:<22}{recall_at(sub_r, sub_p, 1):>10.3f}"
              f"{recall_at(sub_r, sub_p, 3):>10.3f}   (n={len(idx)})")

    # One SUPPORT and one CONTRADICT example (first of each in dev order).
    for lab in ("SUPPORT", "CONTRADICT"):
        i = next(i for i, p in enumerate(pairs) if p["label"] == lab)
        p, r = pairs[i], bm25_rank[i]
        print("\n" + "=" * 70)
        print(f"Claim: {p['claim']}")
        print(f"Abstract: {p['title']}")
        print(f"Gold label: {lab}   gold rationale sentences: {sorted(p['gold'])}")
        print("Top-3 BM25 sentences:")
        for rank, j in enumerate(r[:3], 1):
            mark = "*" if j in p["gold"] else " "
            print(f"  {rank}.{mark}[s{j}] {p['sents'][j]}")
        print("Why questions:")
        for q in why_questions(p["claim"]):
            print(f"  - {q}")
    print("\n(* = gold rationale sentence)")


if __name__ == "__main__":
    main()
