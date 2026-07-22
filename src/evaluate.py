# RAG 평가:
#   1) 검색 성능 - recall@k (질문과 같은 pubid 문단이 top-k에 있는지)
#   2) 최종 답변 정확도 - 생성된 yes/no/maybe vs 정답

import argparse
import json
import os

import numpy as np


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def recall_at_k(retriever, questions, ks=(1, 3, 5, 10)):
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    for q in questions:
        got = retriever.retrieve(q["question"], k=kmax)
        gold_ranks = [r["rank"] for r in got if r["pubid"] == q["pubid"]]
        first = min(gold_ranks) if gold_ranks else 10**9
        for k in ks:
            if first < k:
                hits[k] += 1
    n = len(questions)
    return {k: hits[k] / n for k in ks}


def answer_accuracy(pipeline, questions):
    labels = ["yes", "no", "maybe"]
    correct = 0
    cm = {a: {b: 0 for b in labels} for a in labels}
    preds = []
    for q in questions:
        res = pipeline.answer(q["question"])
        pred = res["decision"]
        gold = q["decision"]
        preds.append({"qid": q["qid"], "gold": gold, "pred": pred,
                      "raw": res["raw"]})
        if gold in labels and pred in labels:
            cm[gold][pred] += 1
        if pred == gold:
            correct += 1
    return correct / len(questions), cm, preds


def plot_all(recall, cm, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # recall@k
    ks = sorted(recall)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([str(k) for k in ks], [recall[k] for k in ks])
    ax.set_ylim(0, 1)
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_title("Retrieval recall@k")
    for i, k in enumerate(ks):
        ax.text(i, recall[k], f"{recall[k]:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recall_at_k.png"), dpi=150)

    # confusion matrix
    labels = ["yes", "no", "maybe"]
    mat = np.array([[cm[a][b] for b in labels] for a in labels], dtype=float)
    norm = mat / np.clip(mat.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title("Answer decision confusion")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "answer_confusion.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="results")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0,
                        help=">0이면 그 개수만 평가 (빠른 확인용)")
    parser.add_argument("--backend", default="st", choices=["st", "hash"])
    parser.add_argument("--gen", default="claude",
                        choices=["claude", "hf", "echo"])
    args = parser.parse_args()

    from retriever import Retriever, HashEmbedder, STEmbedder
    from generator import (RAGPipeline, HFGenerator, EchoGenerator,
                           ClaudeGenerator)

    embedder = HashEmbedder() if args.backend == "hash" else STEmbedder()
    retriever = Retriever.build(
        os.path.join(args.data, "corpus.jsonl"), embedder)
    gen_map = {"claude": ClaudeGenerator, "hf": HFGenerator,
               "echo": EchoGenerator}
    generator = gen_map[args.gen]()
    pipeline = RAGPipeline(retriever, generator, k=args.k)

    test = load_jsonl(os.path.join(args.data, "questions_test.jsonl"))
    if args.limit:
        test = test[:args.limit]

    print(f"검색 백엔드: {retriever.index.backend} / 평가 질문 {len(test)}개")
    recall = recall_at_k(retriever, test)
    acc, cm, preds = answer_accuracy(pipeline, test)

    print("recall@k:", {k: round(v, 4) for k, v in recall.items()})
    print(f"answer accuracy: {acc:.4f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"recall_at_k": recall, "answer_accuracy": acc,
                   "n_test": len(test)}, f, indent=2)
    with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    plot_all(recall, cm, args.out)
    print("저장 완료:", args.out)


if __name__ == "__main__":
    main()
