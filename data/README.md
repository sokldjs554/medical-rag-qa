# data

PubMedQA는 코드에서 자동으로 다운로드됩니다 (HuggingFace datasets).

```
python src/prepare_data.py
```

실행하면 `data/processed/`에 corpus.jsonl, questions_dev.jsonl,
questions_test.jsonl, meta.json이 생성됩니다.

- 데이터셋: https://huggingface.co/datasets/qiaojin/PubMedQA (pqa_labeled)
- 공개 데이터라 별도 로그인/인증 불필요
