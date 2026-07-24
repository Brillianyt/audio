# 项目记忆 — Keyword Detect (Query-by-Example KWS)

## 最佳模型

| 模型 | Dev Seen AUC | 线上 | 架构 |
|------|-------------|------|------|
| AT v8 ep29 | 0.8462 | 0.84 | Whisper base + CharBiGRU + 不确定性融合 |
| LightGBM Stacker | 0.8466 | 待提交 | AT + Fusion + AA 概率级 Stacking |
| Fusion v2 ep9 | 0.8379 | — | Cross-attention 融合，冻结编码器 |

## 数据核心洞察

1. **Dev Unseen 不可信** — enroll_txt 与音频内容 90%+ 不匹配。所有 unseen 评估无效。
2. **测试集规律** — `query_audio = enroll_audio + noise` 或 `f(enroll_audio)`
3. **负样本质量 >> 数量** — 在线交叉配对（1:1）远好于 CSV 预设负样本（26:1）
4. **数据天花板** — 不同架构最终都收敛在 0.83-0.84，50 万对训练数据已用尽

## 模型关键经验

### AT 模型
- **不确定性融合** — AT 内部已做 AA+AT 融合（w_t/w_a ≈ 4:1），额外 AA 无提升
- **简单 > 复杂** — 线性加权 `w_a*sim_a + w_t*sim_t + bias` 优于 cross-attention
- **冻结编码器** — 2.4M 可训参数比 15M 全微调更稳
- **编码器大小不重要** — Whisper small (241M) 不如 base (72M)，瓶颈不在音频编码器

### 噪声增强
- **多样性 > 强度** — burst/lowpass/clip/tail/babble 混合优于纯高斯
- **SpecAugment** — 频率/时间掩码辅助有效

### 集成学习
- **LightGBM > MLP** — 树模型的条件规则更适合少数派正确场景
- **相同数据训练的模型集成无意义** — 错误模式高度重合
- **极值特征有效** — `max_p`, `min_p`, `median_p`, `max-min` 帮树模型捕获分歧信号

### 音频编码器对比

| 编码器 | 参数量 | tough 样本 cos 均值 | 适用性 |
|--------|--------|---------------------|--------|
| Whisper base | 72M | ~0.40 (AT 失败) | AT 文本匹配 |
| **WavLM** | 94M | **0.67** | 声学匹配 |
| **HuBERT** | 94M | **0.63** | 声学匹配 |
| ECAPA-TDNN | 21M | 0.11 | 声纹（不适合） |

### QbyOmni 架构精华（论文 2606.10365v1）

**核心设计**：
- **多模态编码器**：XLS-R (audio) + DistilBERT (text) + G2P (phoneme)，全部冻结
- **全局语境分支**：concat + Self-Attention + GRU → 全局特征
- **关键帧分支**：CTC Selector → 全局唯一去重 → 滑窗平均
- **注意力聚合**：逐向量余弦相似度 → Softmax(×10) 加权全局特征 → 求和融合

**实现要点**：
- CTC 输入用 **原始特征**（无 PE/ME），ID=0 留给 blank
- G2P 在 **DataLoader 预处理**，不在模型 forward 里跑 CPU
- 余弦相似度用 **逐向量归一化**（F.normalize + bmm），不用 Frobenius 范数
- Softmax 温度缩放 **× 10.0**，防止注意力过于平坦
- 关键帧去重用 **全局 seen 集合**（论文的 distinct-token constraint）
- Modality Dropout 在 **PE+ME 之后**做全特征清零

## 文件清单

| 文件 | 用途 |
|------|------|
| `train_dual.py` | AT 训练（Whisper/WavLM/HuBERT 编码器） |
| `train_fusion.py` | 交叉注意力融合模型 |
| `train_qbyomni.py` | QbyOmni 多模态 KWS |
| `stack_ensemble.py` | LightGBM/CatBoost Stacking |
| `lgb_submit.py` | LightGBM 提交脚本 |
| `output/dual_at_v8_text/` | AT v8 训练输出 |
| `output/backup_final/` | 最佳 checkpoint 备份 |
| `submission_lgb.csv` | LightGBM 集成提交 |
| `submission_ep29.csv` | AT v8 ep29 提交 |
