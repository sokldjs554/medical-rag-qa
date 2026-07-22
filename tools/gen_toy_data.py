# PubMedQA 없이 파이프라인 동작만 확인하기 위한 합성 의료 QA 생성기.
# 실제 실험은 반드시 진짜 PubMedQA로 할 것. 이건 smoke test 전용.

import argparse
import json
import os
import random

TOPICS = [
    ("aspirin", "cardiovascular risk", "significant", "yes"),
    ("vitamin D", "bone density", "associated", "yes"),
    ("caffeine", "sleep quality", "no significant", "no"),
    ("statin", "cholesterol level", "significant", "yes"),
    ("screen time", "myopia", "associated", "yes"),
    ("probiotic", "IBS symptoms", "no difference", "no"),
    ("meditation", "blood pressure", "mixed", "maybe"),
    ("omega-3", "depression", "mixed", "maybe"),
]


def make(out_dir, seed=0, per_topic=6):
    rng = random.Random(seed)
    corpus, questions = [], []
    for ti, (drug, outcome, signal, decision) in enumerate(TOPICS):
        pubid = f"P{ti:03d}"
        for j in range(per_topic):
            txt = (f"In this study of {drug}, we examined {outcome}. "
                   f"The analysis found {signal} effect on {outcome} "
                   f"across {rng.randint(50, 900)} participants.")
            corpus.append({"pid": f"{pubid}_{j}", "pubid": pubid,
                           "section": "RESULTS", "text": txt})
        questions.append({
            "qid": pubid, "pubid": pubid,
            "question": f"Is {drug} associated with {outcome}?",
            "decision": decision, "long_answer": f"{signal} effect."})

    rng.shuffle(questions)
    os.makedirs(out_dir, exist_ok=True)
    _dump(os.path.join(out_dir, "corpus.jsonl"), corpus)
    _dump(os.path.join(out_dir, "questions_test.jsonl"), questions)
    _dump(os.path.join(out_dir, "questions_dev.jsonl"), questions)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"n_passages": len(corpus), "n_questions": len(questions)},
                  f, indent=2)
    print(f"합성 데이터: 문단 {len(corpus)} / 질문 {len(questions)} -> {out_dir}")


def _dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
