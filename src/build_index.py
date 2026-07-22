# 코퍼스 문단을 임베딩해서 .npy로 저장 (검색 인덱스 캐시)
# evaluate.py가 이 캐시를 재사용하면 매번 임베딩하지 않아도 됨.

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--backend", default="st", choices=["st", "hash"])
    parser.add_argument("--model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    args = parser.parse_args()

    from retriever import HashEmbedder, STEmbedder

    corpus_path = os.path.join(args.data, "corpus.jsonl")
    corpus = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
    texts = [c["text"] for c in corpus]
    print(f"코퍼스 문단 {len(texts)}개 임베딩 중...")

    embedder = HashEmbedder() if args.backend == "hash" else STEmbedder(args.model)
    emb = embedder.encode(texts)

    out = os.path.join(args.data, "corpus_emb.npy")
    np.save(out, emb)
    print(f"저장: {out}  shape={emb.shape}")


if __name__ == "__main__":
    main()
