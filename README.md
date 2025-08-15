# KillSwitch AI — Prompt Safety Gate + GPT (Streamlit Demo)

## 로컬 실행

    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    # (로컬 가중치면)
    export PAIR_CKPT_PATH=/content/drive/MyDrive/killswitch_ai_demo_zero/models/killswitch_ai_demo_zero_1.pt
    # (클라우드/HF Hub면)
    export HF_TOKEN=hf_...
    export HF_REPO_ID=YOUR_ID/killswitch-pairrank
    export HF_CKPT_FILENAME=killswitch_ai_demo_zero_1.pt

    streamlit run st_app.py