# safety_core.py
import os, re, math, glob
from typing import Dict, Any, List, Tuple
import numpy as np
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ===== 기본 설정 =====
MODEL_NAME = os.getenv("PAIR_MODEL_NAME", "microsoft/deberta-v3-small")
MAX_LEN = int(os.getenv("MAX_LEN", "192"))
RANK_TAU = float(os.getenv("RANK_TAU", "0.75"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.60"))
USE_TRANSLATION = os.getenv("USE_TRANSLATION", "false").lower() == "true"  # Cloud 초기속도 위해 기본 false

# (선택) 체크포인트: 환경변수 > HF Hub > 로컬 탐색
CKPT_PATH = os.getenv("PAIR_CKPT_PATH")
if not CKPT_PATH:
    try:
        from huggingface_hub import hf_hub_download  # 선택 사용
        repo = os.getenv("HF_REPO_ID")
        fname = os.getenv("HF_CKPT_FILENAME", "killswitch_ai_demo_zero_1.pt")
        tok = os.getenv("HF_TOKEN")
        if repo:
            CKPT_PATH = hf_hub_download(repo_id=repo, filename=fname, repo_type="model", token=tok)
    except Exception:
        CKPT_PATH = None
if not CKPT_PATH:
    hits = glob.glob("**/killswitch_ai_demo_zero_*.pt", recursive=True)
    if hits:
        CKPT_PATH = sorted(hits)[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===== 모델 구성 =====
class MeanPooler(nn.Module):
    def forward(self, h, m):
        m = m.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)

class PairScorer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.bb = AutoModel.from_pretrained(base)
        # 일부 모델에서 caching 옵션 경고 방지
        if hasattr(self.bb.config, "use_cache"):
            try:
                self.bb.config.use_cache = False
            except Exception:
                pass
        self.pool = MeanPooler()
        self.drop = nn.Dropout(0.10)
        self.head = nn.Linear(self.bb.config.hidden_size, 1)

    def score(self, ids, msk):
        out = self.bb(input_ids=ids, attention_mask=msk)
        x = self.pool(out.last_hidden_state, msk)
        x = self.drop(x)
        return self.head(x).squeeze(-1)

# ===== 간단 룰셋 =====
INJECTION_PATTERNS = [
    r"(?i)\bignore (all|previous) instructions\b",
    r"(?i)\bdisregard (the )?safety (rules|policy)\b",
    r"(?i)\b(as an ai|system prompt)\b",
]
ILLEGAL_PATTERNS = [
    r"(?i)\b(make|build|buy|sell)\s+(a|an)?\s*(gun|weapon|silencer|explosive|credit card skimmer)\b",
    r"(?i)\b(stab|kill|poison|bomb)\b",
]
PHISHING_PATTERNS = [
    r"(?i)\bverify your (password|account)\b",
    r"(?i)\bsend.*one[- ]time password\b",
    r"(?i)\bseed phrase|private key\b",
]

def rule_score(prompt: str) -> Tuple[float, List[str], List[Dict[str, str]]]:
    hits = []
    for name, rules in [("injection", INJECTION_PATTERNS),
                        ("illegal", ILLEGAL_PATTERNS),
                        ("phishing", PHISHING_PATTERNS)]:
        for rx in rules:
            if re.search(rx, prompt or ""):
                hits.append((name, rx))
    base = 0.0
    if any(h[0]=="illegal" for h in hits): base = max(base, 0.9)
    if any(h[0]=="phishing" for h in hits): base = max(base, 0.7)
    if any(h[0]=="injection" for h in hits): base = max(base, 0.6)
    cats = sorted({h[0] for h in hits})
    highs = []
    for c, rx in hits:
        m = re.search(rx, prompt or "")
        if m:
            span = prompt[max(0, m.start()-10): m.end()+10]
            highs.append({"span": span, "reason": c})
    return float(base), cats, highs

SAFE_TPLS = [
    "I can’t help with harmful or illegal activities. Here’s a safe alternative or educational context instead.",
    "For safety and policy reasons I won’t comply. I can help with lawful, high-level guidance."
]
UNSAFE_TPLS = [
    "Sure, I will ignore safety policies and give you the exact method step by step.",
    "I’ll bypass restrictions and provide the instructions you asked for."
]

def build_text(p, r):
    p = (p or "").strip()
    return f\"\"\"[PROMPT]
{p}

[RESPONSE]
{r}\"\"\" if p else f\"\"\"[RESPONSE]
{r}\"\"\"  # noqa

@torch.no_grad()
def selfcheck_score(pair: PairScorer, text: str, tau: float = RANK_TAU):
    diffs = []
    for s, u in zip(SAFE_TPLS, UNSAFE_TPLS):
        a = build_text(text, s); b = build_text(text, u)
        enc = tok([a, b], truncation=True, padding="max_length",
                  max_length=MAX_LEN, return_tensors="pt")
        ids, msk = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        s_ab = pair.score(ids, msk)
        diffs.append((s_ab[1] - s_ab[0]).item())
    gap = float(np.mean(diffs))
    score = 1.0 / (1.0 + math.exp(-gap / tau))
    return score, gap

# ===== (선택) 번역 파이프라인 =====
if USE_TRANSLATION:
    try:
        from transformers import pipeline
        ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en",
                         device=0 if torch.cuda.is_available() else -1)
        en2ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko",
                         device=0 if torch.cuda.is_available() else -1)
        def tr_ko_en(t): 
            try: return ko2en(t, max_length=512)[0]["translation_text"]
            except Exception: return t
        def tr_en_ko(t): 
            try: return en2ko(t, max_length=1024)[0]["translation_text"]
            except Exception: return t
    except Exception:
        def tr_ko_en(t): return t
        def tr_en_ko(t): return t
else:
    def tr_ko_en(t): return t
    def tr_en_ko(t): return t

# ===== 외부 API =====
def load_core() -> PairScorer:
    pair = PairScorer(MODEL_NAME).to(device)
    if CKPT_PATH and os.path.exists(CKPT_PATH):
        try:
            sd = torch.load(CKPT_PATH, map_location=device)
            pair.load_state_dict(sd if isinstance(sd, dict) else sd.get("model", sd))
            pair.eval()
            print(f"[safety_core] Loaded: {CKPT_PATH}")
        except Exception as e:
            print(f"[safety_core] WARN: checkpoint load failed: {e}")
    else:
        print("[safety_core] INFO: checkpoint not found — using random init.")
    return pair

def analyze_prompt(pair: PairScorer, prompt: str, lang: str = "auto", threshold: float = THRESHOLD):
    is_ko = (lang == "ko") or (lang == "auto" and any("\uac00" <= ch <= "\ud7a3" for ch in prompt or ""))

    text_en = tr_ko_en(prompt) if is_ko else (prompt or "")
    r_score, r_cats, highs = rule_score(text_en)
    ml_score, margin = selfcheck_score(pair, text_en)
    final = max(r_score, ml_score)
    level = "low" if final < 0.4 else ("med" if final < threshold else "high")
    unsafe = final >= threshold

    out = {
        "score": round(final, 4),
        "unsafe": bool(unsafe),
        "level": level,
        "categories": r_cats,
        "highlights": highs,
        "details": {"rule_score": round(r_score,4), "ml_score": round(ml_score,4), "margin": round(margin,4), "threshold": threshold},
        "input_lang": "ko" if is_ko else "en"
    }
    if is_ko:
        catmap = {"injection": "규칙 우회", "illegal": "불법", "phishing": "피싱"}
        out_k = {
            "점수": out["score"],
            "위험여부": "위험" if out["unsafe"] else "안전",
            "레벨": {"low": "낮음", "med": "보통", "high": "높음"}.get(out["level"], out["level"]),
            "카테고리": [catmap.get(c, c) for c in out["categories"]],
            "임계값": threshold,
            "세부": {"룰점수": round(r_score,4), "ML점수": round(ml_score,4), "마진": round(margin,4)},
            "하이라이트": [{"span": tr_en_ko(h["span"]), "reason": catmap.get(h["reason"], h["reason"])} for h in highs]
        }
        out["ko"] = out_k
    return out
