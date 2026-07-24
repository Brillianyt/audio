# QbyOmni v2 — 当前模型架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     输入 (Input)                                      │
│                                                                     │
│  Query Audio  │  Enroll Audio  │  Enroll Text  │  Enroll Phoneme   │
│  (波形 16kHz)  │  (波形 16kHz)   │  (字符串)      │  (G2P→音素ID序列)  │
└──────┬────────┴───────┬────────┴──────┬────────┴────────┬───────────┘
       │                │               │                │
       ▼                ▼               ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐
│ XLS-R 300M   │  │ XLS-R 300M   │  │ DistilBERT │  │ G2P Embedding│
│ (冻结)       │  │ (冻结)       │  │ (冻结)     │  │ 可训 64→128  │
│ 1024→128     │  │ 1024→128     │  │ 768→128    │  │              │
└──────┬───────┘  └──────┬───────┘  └─────┬──────┘  └──────┬───────┘
       │                 │                │                │
       │    Eq_raw       │  Es_raw        │   Et_raw       │  Ep_raw
       │  (B,Tq,128)     │  (B,Ts,128)    │  (B,Lt,128)    │  (B,Lp,128)
       │                 │                │                │
       ▼                 ▼                ▼                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   ├── CTC Selector ───┤                              │
│                                                                      │
│  Eq_raw ──→ Linear(128→42) ──→ ctc_logits_q  (B, Tq, 42)           │
│  Es_raw ──→ Linear(128→42) ──→ ctc_logits_s  (B, Ts, 42)           │
│                                                                      │
│  ctc_logits_q → argmax → 去blank+去重 → 滑窗均值 → Kq (B,Tp,128)    │
│                                                                      │
└───────────────────┬───────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│           加 Positional Encoding + Modality Embedding                │
│       (Eq: Ê = E + PE + v_mod)                                      │
│                                                                      │
│  Eq, Es, Et, Ep  (全部 128 维统一空间)                                │
└───────────────────┬───────────────────────────────────────────────────┘
                    │
        ┌───────────┼──────────────┐
        ▼           │              ▼
┌──────────────┐    │    ┌──────────────┐
│ Global Branch│    │    │ Matching     │
│ (QbyOmni)    │    │    │ (QbyKeyframe)│
│              │    │    │              │
│ [Eq;Es]→SA(2 │    │    │ Kq           │
│ 层)→GRU(64)  │    │    │  │           │
│ →FC→Fa      │    │    │  ▼           │
│ (B,Ts,128)   │    │    │ Cosine Sim   │
│              │    │    │ M_a:Kq vs Es │
│ [Eq;Ep]→同理  │    │    │ M_p:Kq vs Ep│
│ →Fp (B,Lp,D) │    │    │ M_t:Kq vs Et│
│              │    │    │  │           │
│ [Eq;Et]→同理  │    │    │  ▼           │
│ →Ft (B,Lt,D) │    │    │ Softmax(M*10)│
└──────┬───────┘    │    │   × Fa/Fp/Ft │
       │            │    │   → Ka,Kp,Kt │
       │            │    │  (B,Tp,128)  │
       ▼            │    └──────┬───────┘
┌──────────────┐    │           │
│frame_cls_phone│   │           │
│frame_cls_text │   │           │
│ (Eq.8监督)    │   │           │
└──────────────┘    │           │
                    ▼           ▼
          ┌─────────────────────────┐
          │  Fusion: Ka + Kp + Kt   │
          │  → Mean Pool (B,128)    │
          │  → Classifier(128→64→1) │
          │  → Sigmoid → score      │
          └──────────┬──────────────┘
                     ▼
               [0, 1] 置信度

```

## 损失函数 (Eq.10)

```
L_total = L_u + L_s^p + L_s^t + 0.2 * L_c

L_u       = BCE(score, label)                  — Eq.7 主匹配损失
L_s^p     = 0.05 * frame_BCE(phone_feats)      — Eq.8 音素级帧监督
L_s^t     = 0.05 * frame_BCE(text_feats)       — Eq.8 文本级帧监督
L_c       = CTC(ctc_logits_q, query_phonemes)  — Eq.9 查询音频CTC
          + CTC(ctc_logits_s, enroll_phonemes)  — Eq.9 注册音频CTC
```

## 训练配置（当前）

| 参数 | 值 | 说明 |
|------|----|------|
| bs | 16 | 实际 batch |
| grad_accum | 32 | 有效 batch = 512 |
| lr | 1e-3 | AdamW |
| 可训参数 | ~0.7M | 冻结 XLS-R + DistilBERT |
| 每 epoch | ~1030 batch | 1500 正 + 15000 负 |
| 优化步/epoch | ~32 | 1030/32 |
