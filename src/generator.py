# 검색된 문단을 근거로 yes/no/maybe 판정 + 근거를 생성하는 LLM 래퍼
#
# 기본은 무료 오픈모델(HuggingFace instruct 모델)을 Colab GPU에서 실행.
# LLM 없이 파이프라인만 확인하고 싶을 때를 위해 EchoGenerator도 둠.

import re

DECISIONS = ["yes", "no", "maybe"]

PROMPT_TMPL = """You are a biomedical question answering assistant.
Use ONLY the provided research abstract excerpts to answer the question.
Answer with exactly one of: yes, no, maybe. Then give a one-sentence reason.

Context:
{context}

Question: {question}

Answer (yes/no/maybe) and reason:"""


def build_prompt(question, passages, max_chars=1800):
    ctx = ""
    for p in passages:
        piece = f"- {p['text'].strip()}\n"
        if len(ctx) + len(piece) > max_chars:
            break
        ctx += piece
    return PROMPT_TMPL.format(context=ctx.strip(), question=question.strip())


def parse_decision(text):
    # 생성문에서 yes/no/maybe를 뽑아냄. 앞쪽 단어를 우선.
    low = text.strip().lower()
    m = re.match(r"[^a-z]*\b(yes|no|maybe)\b", low)
    if m:
        return m.group(1)
    for d in DECISIONS:                 # 앞에서 못 찾으면 전체에서 첫 등장
        if re.search(rf"\b{d}\b", low):
            return d
    return "maybe"                       # 못 찾으면 보수적으로 maybe


class HFGenerator:
    """HuggingFace instruct 모델. Colab 무료 T4에 올라가는 소형 모델 권장."""

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device=None,
                 max_new_tokens=64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None)
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def generate(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


class ClaudeGenerator:
    """Anthropic Claude API 백엔드.

    공고 기술스택의 'LLM API'에 해당. 소형 오픈모델보다 판정/근거 품질이 좋고
    GPU가 필요 없음(임베딩만 로컬, 생성은 API).
    API 키는 환경변수 ANTHROPIC_API_KEY에서 읽음 — 코드/깃에 절대 하드코딩 금지.
    """

    def __init__(self, model="claude-3-5-haiku-latest", max_tokens=64):
        from anthropic import Anthropic
        self.client = Anthropic()          # ANTHROPIC_API_KEY 환경변수 사용
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt):
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # content 블록에서 텍스트만 모음
        return "".join(
            b.text for b in msg.content if getattr(b, "type", "") == "text"
        ).strip()


class EchoGenerator:
    """LLM 없이 동작 확인용. 근거 문단의 표면 신호로 대충 판정."""

    def generate(self, prompt):
        low = prompt.lower()
        neg = sum(low.count(w) for w in ["no significant", "not associated",
                                         "no difference", "did not"])
        pos = sum(low.count(w) for w in ["significant", "associated",
                                         "increased", "effective"])
        if pos > neg:
            return "yes. (echo) context suggests a positive association."
        if neg > pos:
            return "no. (echo) context suggests no association."
        return "maybe. (echo) evidence is mixed."


class RAGPipeline:
    def __init__(self, retriever, generator, k=5):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def answer(self, question):
        passages = self.retriever.retrieve(question, k=self.k)
        prompt = build_prompt(question, passages)
        raw = self.generator.generate(prompt)
        return {
            "question": question,
            "decision": parse_decision(raw),
            "raw": raw,
            "retrieved": passages,
        }
