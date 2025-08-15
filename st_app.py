import os
import streamlit as st
from openai import OpenAI
from safety_core import load_core, analyze_prompt

# 기본값(Secrets 미설정 대비)
os.environ.setdefault("THRESHOLD", "0.60")
os.environ.setdefault("USE_TRANSLATION", "false")   # 초기속도 위해 기본 false
os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="KillSwitch + GPT", page_icon="🛡️")
st.title("🛡️ KillSwitch + GPT — Streamlit 데모")

# ── 모델은 '필요할 때' 한 번만 로드(초기 흰 화면 방지) ──
@st.cache_resource
def _load_pair():
    return load_core()

if "pair" not in st.session_state:
    st.session_state.pair = None  # 최초엔 미로딩

with st.sidebar:
    st.subheader("설정")
    openai_key = st.text_input(
        "OPENAI_API_KEY", type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Streamlit Secrets에 저장해두면 자동으로 채워집니다."
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.text_input("OpenAI 모델", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="model_name")
    threshold = st.slider("임계값", 0.0, 1.0, float(os.getenv("THRESHOLD", "0.60")), 0.01)
    lang = st.selectbox("입력 언어", ["auto", "ko", "en"], index=0)
    allow_unsafe = st.checkbox("위험해도 GPT 호출 강행", value=False)

prompt = st.text_area("프롬프트", height=180, placeholder="예) 초등 과학 프로젝트 아이디어 5개 알려줘.")
go = st.button("분석 후 GPT 호출")

if go:
    if not prompt.strip():
        st.warning("프롬프트를 입력하세요.")
        st.stop()

    # 최초 클릭 시 한 번만 모델 로드
    if st.session_state.pair is None:
        with st.spinner("모델 불러오는 중... (최초 1회)"):
            st.session_state.pair = _load_pair()

    with st.spinner("안전 점검 중..."):
        analysis = analyze_prompt(st.session_state.pair, prompt, lang=lang, threshold=threshold)

    if analysis["unsafe"] and not allow_unsafe:
        st.error("위험도가 높아 GPT 호출을 차단했어요.")
        st.json(analysis.get("ko", analysis))
        st.stop()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY 필요(좌측 사이드바 입력 또는 Secrets 저장).")
        st.json(analysis.get("ko", analysis))
        st.stop()

    client = OpenAI(api_key=key)
    sys_hint = "답변은 한국어로 간결하고 안전하게." if analysis.get("input_lang") == "ko" else "Answer concisely and safely."
    full_prompt = f"{sys_hint}\n\n[사용자]\n{prompt}"

    with st.spinner("GPT 호출 중..."):
        try:
            resp = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                input=full_prompt,
            )
            output_text = resp.output_text
        except Exception as e:
            st.error(f"OpenAI 호출 실패: {e}")
            output_text = None

    if output_text:
        st.success("완료")
        st.subheader("GPT 응답")
        st.write(output_text)

    st.subheader("분석 결과")
    st.json(analysis.get("ko", analysis))
