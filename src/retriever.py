# 임베딩 + 벡터 검색기
#
# 임베딩 백엔드 2가지:
#   - "st": sentence-transformers (실제 실험용, Colab에서 사용)
#   - "hash": 의존성 없는 해싱 임베딩 (스모크 테스트용, 성능은 낮음)
# 벡터 검색 백엔드 2가지:
#   - FAISS가 있으면 FAISS 사용
#   - 없으면 numpy 코사인 유사도로 폴백 (소규모에서는 충분)

import json

import numpy as np


def l2_normalize(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


class HashEmbedder:
    """단어를 해싱해 bag-of-words 벡터로 만드는 초경량 임베더.

    실제 의미를 담지는 못하지만, 파이프라인이 도는지 확인하는 용도로 충분.
    실제 실험은 STEmbedder(sentence-transformers)를 쓸 것.
    """

    def __init__(self, dim=256):
        self.dim = dim

    def encode(self, texts, batch_size=256, show_progress=False):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                vecs[i, hash(tok) % self.dim] += 1.0
        return l2_normalize(vecs)


class STEmbedder:
    """sentence-transformers 임베더. 의료 도메인 모델을 기본값으로."""

    def __init__(self, model_name="pritamdeka/S-PubMedBert-MS-MARCO", device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=64, show_progress=True):
        emb = self.model.encode(
            list(texts), batch_size=batch_size,
            show_progress_bar=show_progress, convert_to_numpy=True,
            normalize_embeddings=True)
        return emb.astype(np.float32)


class VectorIndex:
    """FAISS 있으면 FAISS, 없으면 numpy로 top-k 검색."""

    def __init__(self, embeddings):
        self.emb = embeddings.astype(np.float32)
        self.backend = "numpy"
        self._faiss = None
        try:
            import faiss
            index = faiss.IndexFlatIP(self.emb.shape[1])  # 정규화 후 IP=코사인
            index.add(self.emb)
            self._faiss = index
            self.backend = "faiss"
        except Exception:
            pass

    def search(self, queries, k=5):
        q = queries.astype(np.float32)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return idx, scores
        # numpy 폴백
        sims = q @ self.emb.T                     # (nq, N)
        idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        # 상위 k를 점수순 정렬
        row = np.arange(len(q))[:, None]
        order = np.argsort(-sims[row, idx], axis=1)
        idx = idx[row, order]
        scores = sims[row, idx]
        return idx, scores


class Retriever:
    def __init__(self, corpus, embedder, index):
        self.corpus = corpus            # list of {pid, pubid, text, ...}
        self.embedder = embedder
        self.index = index

    @classmethod
    def build(cls, corpus_path, embedder, emb_cache=None):
        corpus = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
        if emb_cache is not None:
            emb = np.load(emb_cache)
        else:
            emb = embedder.encode([c["text"] for c in corpus])
        return cls(corpus, embedder, VectorIndex(emb))

    def retrieve(self, question, k=5):
        qv = self.embedder.encode([question], show_progress=False)
        idx, scores = self.index.search(qv, k=k)
        out = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0])):
            c = self.corpus[int(i)]
            out.append({**c, "score": float(s), "rank": rank})
        return out
