# PubMedQA(pqa_labeled) -> RAG용 코퍼스 + 질문셋으로 변환
#
# PubMedQA는 질문마다 관련 의학 논문 초록(context)이 붙어 있고,
# yes/no/maybe 정답(final_decision)이 달려 있음.
# RAG 관점에서는:
#   - 모든 질문의 context 문단을 모아 "검색 대상 코퍼스"로 씀
#   - 각 문단이 어느 논문(pubid)에서 왔는지 태깅해두면,
#     질문과 같은 pubid 문단이 검색되는지로 retrieval 성능을 잴 수 있음

import argparse
import json
import os


def load_pubmedqa():
    # datasets 라이브러리로 로드 (Colab/로컬에 datasets 필요)
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    return ds


def build(ds, out_dir, seed=42, test_size=500):
    corpus = []          # {pid, pubid, text}
    questions = []       # {qid, pubid, question, decision, long_answer}
    seen_passage = set()

    for i, row in enumerate(ds):
        pubid = str(row.get("pubid", i))
        ctx = row["context"]
        passages = ctx["contexts"] if isinstance(ctx, dict) else ctx
        labels = ctx.get("labels", []) if isinstance(ctx, dict) else []

        for j, text in enumerate(passages):
            text = (text or "").strip()
            if not text:
                continue
            # 같은 문단이 중복 저장되지 않게
            key = (pubid, j)
            if key in seen_passage:
                continue
            seen_passage.add(key)
            section = labels[j] if j < len(labels) else ""
            corpus.append({
                "pid": f"{pubid}_{j}",
                "pubid": pubid,
                "section": section,
                "text": text,
            })

        questions.append({
            "qid": pubid,
            "pubid": pubid,
            "question": row["question"].strip(),
            "decision": row["final_decision"].strip().lower(),
            "long_answer": row.get("long_answer", "").strip(),
        })

    # 질문셋을 dev/test로 분할 (코퍼스는 공통, 재현 위해 seed 고정 정렬 셔플)
    import random
    rng = random.Random(seed)
    rng.shuffle(questions)
    test = questions[:test_size]
    dev = questions[test_size:]

    os.makedirs(out_dir, exist_ok=True)
    _dump(os.path.join(out_dir, "corpus.jsonl"), corpus)
    _dump(os.path.join(out_dir, "questions_dev.jsonl"), dev)
    _dump(os.path.join(out_dir, "questions_test.jsonl"), test)

    meta = {
        "n_passages": len(corpus),
        "n_questions": len(questions),
        "n_dev": len(dev),
        "n_test": len(test),
        "decision_dist": _dist([q["decision"] for q in questions]),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("코퍼스 문단 수:", len(corpus))
    print("질문 수:", len(questions), "(dev", len(dev), "/ test", len(test), ")")
    print("정답 분포:", meta["decision_dist"])
    print("저장 완료:", out_dir)


def _dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dist(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=500)
    args = parser.parse_args()

    ds = load_pubmedqa()
    build(ds, args.out, seed=args.seed, test_size=args.test_size)


if __name__ == "__main__":
    main()
