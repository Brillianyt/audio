# Keyword Detect — 语音关键词检测 (Query-by-Example KWS)

## 项目概述

本项目是**科大讯飞 AI 开发者大赛**的语音关键词检测任务解决方案。任务属于基于语音注册的关键词检测 (Query-by-Example Keyword Spotting)：给定一段注册音频和对应文本，判断测试音频是否包含相同文本的关键词。

项目采用**双编码器架构 (Dual-Encoder)**，独立训练两个子模型后在推理时加权集成：

- **Model A (Audio-Audio)**：注册音频与测试音频的声学匹配，使用 Whisper/WavLM 编码器 + Supervised Contrastive Loss
- **Model B (Audio-Text)**：注册文本与测试音频的跨模态匹配，使用 Whisper 音频编码 + 字符/音素文本编码 + BCE Loss

此外还包含多个独立探索的基线模型（纯 CNN、Whisper-only、WavLM-only）。

### 比赛背景

- 评价指标：AUC (macro-average)，基于唤醒后验概率计算
- 数据规模：训练集 50 万对 (8357 词)，开发集 1 万对 (2247 词, ~40-45% 集外词)，验证集 10 万对 (3619 词, ~40% 集外词)
- 测试场景：-10 ~ 5 dB 信噪比 + 发音相似词混淆（如 hi <--> haier）
- 允许使用外部开源训练数据和开源模型权重（但不可直接用外部模型做推理判决）

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.0+, torchaudio |
| 音频编码器 | OpenAI Whisper (base) |
| 文本编码器 | Char-level BiGRU |
| 数据处理 | soundfile, zipfile (从 zip 读取 WAV), numpy |
| 评估指标 | scikit-learn roc_auc_score |
| 依赖管理 | requirements.txt (pip) |
| 音频增强 | Gaussian noise, SpecAugment |


## 核心模型架构

### 1. 双编码器集成架构 (`train_dual.py`)

这是项目的主模型，训练两个独立模型后在推理时加权集成：

**Model A — Audio-Audio (SupCon)**:
- 编码器：Whisper base (部分解冻最后 N 层) 或 WavLM base-plus
- 池化层：Attentive Statistics Pooling (ASP) — 加权均值 + 加权标准差
- 投影层：2×dim → embed_dim (256)
- 损失函数：Supervised Contrastive Loss (温度 0.25)
- 辅助损失：负样本余弦中心化惩罚 (push neg cos < 0)
- 增强：每步随机高斯噪声 (-10~5 dB SNR)

**Model B — Audio-Text (BCE)**:
- 预处理：前后静音填充
- 音频编码器：Whisper base (部分解冻，与 Model A 共享结构)
- 文本编码器：CharBiGRUEncoder (28 字符嵌入 + 双向 GRU)
- 匹配分数：`cos(enroll_text, query_audio)` + `cos(enroll_audio, enroll_text)` 对齐
- 损失函数：BCEWithLogitsLoss (pos_weight=5.0) + margin loss + MSE alignment
- 学习率分组：Whisper params lr/5，其他 lr=1e-3


### 2. Whisper V3 基线 (`baseline/train_whisper_v3.py`)

- **WhisperEncoderV3**：Whisper base 编码器 + ASP Pooling + 投影
- **MultiTaskWhisperKWSV3**：孪生架构 (共享编码器)，余弦相似度 + scale/bias → sigmoid
- **AngularPrototypicalLoss**：episodic training (support→prototype→query CE)，提升 unseen 泛化
- **ImprovedPKSampler**：每 epoch 固定 batch 数 (P 词 × K 样本/词)
- **Scheduler**：Warmup → Flat → Linear Decay
- **SpecAugment**：频率/时间掩码
- **WhisperTextKWS**：Whisper 编码器 + Phoneme RoPE Transformer 文本分支，不确定性加权融合


### 4. 孪生 CNN 基线 (`baseline/train.py`, `baseline/model.py`)

- **Encoder**：2 层 Conv2D (32→64 filters) + AdaptiveAvgPool2d + FC → embed_dim
- **SiameseKWS**：共享编码器 + 可学习 scale/bias → logit
- 特征：40 维 log-mel 谱图，固定 100 帧
- 增强：高斯噪声 + SpecAugment

## 训练数据增强策略

### 难负样本挖掘 (Hard Negative Mining)

项目使用了多种难负样本挖掘方法：

| 方法 | 原理 | 文件 |
|------|------|------|
| 字符编辑距离 | 基于 Levenshtein 距离找拼写相似词 | `hard_neg.py` |
| 纯音素编辑距离 | CMU dict → ARPAbet 音素序列 → 编辑距离 | `hard_neg_phoneme.py`, `mine_phoneme.py` |
| Embedding 余弦距离 | Whisper/WavLM 编码音频 → 词级 centroid → 高 cos 对不同词 | `build_hard_neg.py`, `build_hard_neg_wavlm.py` |
| 迭代挖掘 | 用已训练的 Whisper+Text 模型编码 → 找 top-K 近邻 | `iter_hard_neg.py` |

### 自配对 (Self-Paired)

- `gen_pairs.py`：同词不同 enrollment 音频配对 → 正样本；异词交叉配对 → 负样本
- `gen_pairs_external.py`：从外部数据集（如 Speech Commands）扫描 word/*.wav 结构，自动转换

### 数据增强

- 高斯噪声：每步随机 SNR (-10~5 dB)
- SpecAugment：频率/时间掩码 (用于 log-mel 谱图)

## 训练命令

### 双编码器训练

```bash
# Audio-Audio 模型 (SupCon)
python train_dual.py --name aa_v1 --mode audio --epochs 15

# Audio-Text 模型 (BCE)
python train_dual.py --name r2_init --mode text --epochs 15

# 从 checkpoint 恢复
python train_dual.py --name aa_v1 --mode audio --resume

# 指定编码器 (whisper/wavlm)
python train_dual.py --name wavlm_aa --mode audio --encoder wavlm
```


### 基线模型训练

```bash
# 孪生 CNN
python baseline/train.py --epochs 15 --bs 128

# Whisper V3
python baseline/train_whisper_v3.py --name whisper_v3 --epochs 10

# WavLM
python baseline/train_wavlm.py --name wavlm_v1 --epochs 10 --hard-neg
```

### 数据挖掘

```bash
# 自配对
python baseline/gen_pairs.py --csv train/train_label.csv --out train/self_paired.json

# 音素难负样本
python baseline/mine_phoneme.py --csv train/train_label.csv --out baseline/hard_neg_phoneme.json

# Embedding 挖掘
python baseline/build_hard_neg.py \
    --whisper-ckpt output/whisper_v3/best.pt \
    --out baseline/hard_neg_whisper.json

# 外部数据集
python baseline/gen_pairs_external.py --root /path/to/speech_commands
```

## 推理与提交

推理流程（推断测试集）：
1. 从 `eval/eval_seen/wav.zip` 和 `eval/eval_unseen/wav.zip` 读取音频
2. 从 `evalcsv_without_label/` 读取 CSV (id, enroll_txt)
3. 模型输出后验概率 posterior (值域 [0,1])
4. 写入 CSV，格式：`id,posterior`
   - seen 子集 ID 格式：`seen_pair_000001`
   - unseen 子集 ID 格式：`unseen_pair_000001`
5. seen/unseen 合并到一个 CSV 提交

### AUC 本地计算

```bash
python egs/cal_auc.py --csv egs/example.csv
```

本地计算需要 label 列，线上提交不需要 label 列。

## 配置管理

路径配置集中在 `baseline/config.py` 的 `Paths` dataclass：

- 根目录通过 `ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))` 自动定位
- 支持嵌套的 `dev/dev/` 目录检测（解压导致的嵌套）
- 支持 `train_subset/` 优先于 `train/` 的自动降级

各训练脚本有独立的 Config class（如 `WhisperConfigV3`, `WavLMConfig`），内部自动补全路径。

## 训练输出结构

每个实验的输出在 `output/{experiment_name}/` 下：

- `best.pt` — 开发集最佳 AUC 的 checkpoint (含 model state_dict, auc_seen, auc_unseen, epoch)
- `latest.pt` — 最新 epoch 的 checkpoint
- `train.log` — 训练日志
- `cos_dist.jsonl` — 余弦分布统计 (cos_pos/cos_neg 的均值、分位数、直方图)
- `submission.csv` — 推理结果
- `experiment.json` — 实验配置摘要

Checkpoint 格式：
```python
{
    "model": model.state_dict(),
    "auc_unseen": float,
    "auc_seen": float,
    "epoch": int,
    "embed_dim": int,
    "whisper_model": str,  # 仅 Whisper 模型
}
```

## 开发约定

### 编码风格

- 注释以中文为主，夹杂英文技术术语
- 变量命名：snake_case
- 类命名：PascalCase
- 常量：UPPER_SNAKE_CASE
- 类型注解：部分使用 `from __future__ import annotations`

### 数据处理约定

- 所有音频统一为 16kHz 单声道 float32
- 训练音频打包在 ZIP 中，按 `wav/{pair_id}_{role}.wav` 结构存储 (role: enroll/query)
- 过长的音频截断到 `max_audio_sec` (通常 1.5s 或 3s)
- 数据集类支持 `enroll_id` / `query_id` 分离（用于难负样本的跨 pair 配对）
- 使用 per-process zip 缓存 (`os.getpid()` 作为 key) 保证多进程 DataLoader 安全

### 损失函数

- **主模型 (Model A)**：SupCon Loss + 余弦中心化惩罚
- **主模型 (Model B)**：BCEWithLogitsLoss (pos_weight=5.0) + margin losses + MSE alignment
- **Whisper V3**：BCE + AngularPrototypicalLoss (lambda=0.08)
- **WavLM**：BCE + AngularPrototypicalLoss + Phoneme auxiliary loss (lambda=0.2)

### 优化器

- 默认：AdamW (weight_decay=1e-4)
- 学习率分组：预训练编码器参数使用更低 lr (lr/5 或 lr/10)
- AMP：使用 `torch.amp.autocast` 和 `GradScaler` 加速
- 梯度裁剪：max_norm=5.0
- 梯度累积：grad_accum 参数

### 学习率调度

- **Whisper V3**：Warmup → Flat → Linear Decay
- **WavLM**：Cosine Warmup + Cosine Annealing
- **Dual-Encoder**：固定学习率 (3e-4)

## 关键文件对照表

| 操作 | 脚本 | 说明 |
|------|------|------|
| 双编码训练 | `train_dual.py` | 主训练脚本 |
| 集成推理 | `baseline/ensemble_infer.py` | 加载 AA+AT 模型，网格搜索权重 |
| 音频数据读取 | `baseline/data.py` | `read_wav()`, `add_noise()`, `PairDataset` |
| 路径配置 | `baseline/config.py` | `PATHS`, `AUDIO`, `TRAIN` |
| Whisper 编码器 | `baseline/train_whisper_v3.py` | `WhisperEncoderV3`, `WhisperTextKWS` |
| WavLM 编码器 | `baseline/train_wavlm.py` | `WavLMEncoder`, `WavLMKWS` |
| SupCon Loss | `train_dual.py` | `supcon_loss()` |
| Angular Prototypical | `baseline/train_whisper_v3.py` | `AngularPrototypicalLoss` |
| 音素编码器 | 多个文件 | `PhonemeTextEncoder`, `CharBiGRUEncoder` |
| AUC 计算 | `egs/cal_auc.py` | 本地评估脚本 |
| 难负样本 (字符) | `baseline/hard_neg.py` | `HardNegativeMiner` |
| 难负样本 (音素) | `baseline/hard_neg_phoneme.py` | `PhonemeHardNegativeMiner` |

## 注意事项

1. **测试集规范**：绝对不能用测试集参与训练（比赛规则）
2. **外部模型规范**：允许使用开源模型权重做特征提取，但不能直接用外部模型做推理判决
3. **数据安全**：所有数据已脱敏，`.env` 等敏感文件禁止读取
4. **Git LFS**：模型权重 (.bin, .pt, .pth)、音频 (.wav, .zip)、CSV 等大文件用 LFS 管理

## 依赖安装

```bash
pip install -r requirements.txt
pip install openai-whisper     # Whisper 编码器
pip install transformers       # WavLM 模型
pip install cmudict            # CMU 发音词典（离线）
```

建议使用 Python 3.10+ 和 CUDA 12+ 环境。

## 耗时任务执行规范（必须遵守）

所有可能耗时 >30s 的任务必须遵循以下模式：

### 1. 后台启动 + cron 定时通知

```python
# ✅ 正确：后台启动 + cron 定时检查
Bash(command="long_running_training ...", run_in_background=true)
CronCreate(cron="*/10 * * * *", prompt="检查训练进度，读取日志...")

# ❌ 错误：主进程 sleep 阻塞
Bash(command="sleep 300 && check_progress")  # 禁止！
```

### 2. 具体规则

1. **后台启动**：所有训练/推理/大数据处理用 `Bash(run_in_background=true)` 放到后台
2. **不阻塞主进程**：主Agent 不使用 `sleep`、`TaskOutput(block=true)` 或任何轮询等待——后台任务完成时系统会自动通知
3. **cron 定时检查**：需要中间进度时，用 `CronCreate` 创建定时任务，到点自动通知
4. **响应通知**：收到后台任务完成或 cron 通知后，查看结果并决定下一步（调参、重跑、继续）
5. **GPU 共享**：单 GPU 上多个训练可能同时运行，观察显存占用后动态调整 batch_size
6. **失败恢复**：训练中断后检查退出码和日志摘要，判断改代码重跑还是调整参数

### 3. 下载规范

1. **上网找正确方式**：下载模型/数据时，先在网上搜索正确的下载地址和方法（如 HuggingFace 镜像 hf-mirror.com）
2. **确认文件存在**：先 HEAD 请求检查文件是否存在（区分 safetensors / bin / msgpack），不要盲目下载
3. **断点续传**：大文件用 wget -c 支持断点续传
4. **缓存复用**：下载前检查本地缓存 `/root/.cache/huggingface/` 和项目目录下是否已有

## 项目状态与迭代规范（2026-07-19）

### 模型迭代阶段

项目已进入**模型迭代阶段**，核心原则：

1. **使用已有权重**：每次新实验都用上一轮最好的 `output/at_vx/best.pt` 初始化（`--resume`），绝不从 0 训练
2. **所有训练脚本必须支持 `--resume` 参数**：从 `latest.pt` 恢复（含 epoch、optimizer 状态），确保中断不丢进度
3. **持续递进**：AT v2 → v3 → v4 ... 每次只改一个变量，保留历史学习成果
4. **推理结果**：AT v2（unseen=0.7446）已产出 `submission_at_v2.csv`，可提交线上测评

### 有效模型

| 模型 | 架构 | unseen AUC | 状态 |
|------|------|-----------|------|
| AT v2 | V1(CharBiGRU 256d) + BCE + margin loss | **0.7446** | 已完成 |
| AT v3 | V1 + 去重数据 + 32 新词 + 3847 hard neg 词对 | 训练中 | epoch 1/20 |

### 训练数据现状

| 数据源 | 唯一词对数 | 说明 |
|--------|-----------|------|
| 原始 CSV | 500K pairs | 8357 词，赛题提供 |
| Speech Commands v2 | 117 唯一词对 (+32 新词) | 3.3G 本地，394MB zip |
| AT v2 模型难负例 | 3847 唯一词对 | cos>0.2 的发音相似词 |
| 其他 (self_paired, external) | ~171K 唯一负词对 | 已去重 |
| LibriSpeech | 未下载 (磁盘不足) | 50G 磁盘用满，放弃 |

### AT v3 训练核心改进

1. **去重采样**：每个唯一词对最多 1~3 条，防止重复数据导致记忆
2. **每 epoch 重新洗牌**：从去重池中随机选子集，不同 epoch 看到不同组合
3. **新词扩充**：Speech Commands 提供 32 个原词表没有的新词
4. **难负例**：AT v2 模型挖出的 3847 个 embedding 相似词对（如 disperse↔display）
5. **正样本多样性**：每个词保留最多 10 个音频变体

### 训练策略要点

1. **负样本只用 hard neg 训练**：易负例（不同发音的词）cos 本来就低，对梯度无贡献。只有 hard neg（发音相似词如 hi↔haier）能提供有效 cos- 信号。不要加 easy neg 稀释。
2. **AT vx 系列迭代训练**：每次新实验都用上一轮最好的 `output/at_vx/best.pt` 权重初始化（`--resume` 或 `--load-ckpt`），不要从 0 训练。v2→v3 持续递进，不丢历史学习成果。
3. **cos- ≈ 0 在早期是正常的**：发音完全不同的词正交是合理的，不需要强行推到 -1
4. **难例挖掘时机**：等文本编码器学会音素映射后再做（而非训练初期）
5. **去重比扩总量重要**：一个词对重复 100 次不如 100 个不同词对各 1 次
6. **关注 hard neg 的 cos-，而非全局平均**：全局 cos- 被大量易负例稀释
7. **所有耗时任务必须后台运行**：用 CronCreate 设定时通知，不用 sleep

## 最终模型

### AT 最终 — `output/at_v5/at_final.pt`

| 指标 | 值 |
|------|-----|
| 架构 | WhisperEncoder(解冻2层) + PhonemeBiGRUEncoder(CMU音素→Embedding→BiGRU) + ComparisonHead |
| 文本输入 | CMU ARPAbet 音素序列（40符号），通过 `cmudict` 词典转换 |
| 损失函数 | BCEWithLogitsLoss + margin loss + MSE alignment |
| 训练数据 | 60K pos（去重，每词最多30条）+ 300K hard_neg ×10 oversample，无 easy_neg |
| 采样策略 | 每 epoch 随机洗牌采样，360K 样本 |
| 优化器 | AdamW，whisper lr=6e-5，其他 lr=1e-3，cosine warmup |
| 最佳结果 | **unseen AUC = 0.6325 / seen AUC = 0.6109** |

关键成功因素：
- 音素输入替代字符输入，消除了字符→发音的映射学习负担
- BiGRU 建模音素序列的时序依赖关系（发音顺序）
- ComparisonHead 提供强梯度信号，eval 用 cosine 相似度

### AA 最终 — `output/aa_pk/aa_final.pt`

| 指标 | 值 |
|------|-----|
| 架构 | WhisperEncoder(解冻6层) + cosine 相似度 |
| 损失函数 | AngularPrototypicalLoss |
| 训练数据 | PKSampler: P=32 words × K=4 audios per batch，300 batches/epoch |
| 最佳结果 | **seen AUC = 0.7490 / unseen AUC = 0.5031** |

关键成功因素：
- PKSampler：每 batch 32词×4音频，batch内词级别对比
- AngularPrototypicalLoss：episodic 训练模拟注册→测试流程
- 解冻6层 Whisper 提供足够容量记忆 seen 词

### 推理集成策略

```
aa_conf = abs(aa_prob - 0.5) * 2   # 置信度 0~1
at_conf = abs(at_prob - 0.5) * 2
final_prob = (aa_conf * aa_prob + at_conf * at_prob) / (aa_conf + at_conf)
```

AA 对 seen 词自信，AT 对 unseen 自信。各自不自信时对方的权重更高。

### 已探索但放弃的方向

- **Whisper 解冻 + Siamese AA** — 编码器记忆训练词，unseen=0.51
- **Whisper 冻结 + 帧级差异图** — prob+ 0.80 但用户坚持不用 Whisper
- **HuBERT/WavLM + 帧级差异比较** — 自监督特征过于平滑，没有区分度，unseen≈0.49
- **WavLM + 相似度矩阵 + 2D CNN** — seen=0.515, unseen=0.482
- **LibriSpeech 下载** — 磁盘 50G 不足以存放 6G tar.gz + 解压

### 关键经验

1. **AT 和 AA 解决不同问题，不能用同一套数据** — AT 是跨模态泛化（text↔audio），需要模型误判的 hard neg；AA 是纯声学匹配（audio↔audio），不需要负样本
2. **AA 不需要负样本** — 不同词声学天然不匹配，只需学同词帧级对齐。BCE + 负样本反而引入噪声
3. **AT 的 hard neg 必须来自模型实际误判** — 不是文本空间邻居，而是模型真正混淆的 pair。OHEM 逐对跑模型打分才能得到
4. **数据去重比扩量更重要**：唯一词对覆盖比总 pair 数关键
5. **不要占主进程**：所有后台任务用 CronCreate 定时检查，不用 sleep

## 阶段性训练方案 (2026-07-21)

### 最重要发现：AT 正确架构是 CharBiGRU，不是 PhonemeBiGRU

- **/output/backup_final/at_v8_ep22_seen08131_unseen07808（CharBiGRU, 0.7889 unseen）才是最强 AT** — 训练脚本 


### AA 提升来自 SupCon + 降解冻，不是 frame attention

今天 AA 实验汇总：
- AA 纯 cosine: seen=0.76（基线）
- AA Hybrid v2: seen=0.787 — **提升来自 SupCon loss + unfreeze 3，attention 分支全程死亡（α→0.98）**
- AA 纯 cosine + SupCon: seen=0.768 — 确认 SupCon 有效
- Frame cross-attention 三次尝试均失败：v1 max-pooling 饱和、v2 无界输出、v3 死分支

**有效改进**：SupCon 损失 + 降解冻层数（6→2-3）。Frame attention 无贡献。



## 模型诊断与修复 (2026-07-19)

经过逐行审查训练代码，发现以下致命问题并已修复：

### 🔴 P0：AT 训练-评估目标不一致

**问题**：`train_at_v3.py` 训练时优化 ComparisonHead 输出 `logit`，但评估时丢弃 ComparisonHead，改用 raw `cos(et, eq) * 8 → sigmoid`。

**后果**：模型花全部力气优化的 ComparisonHead 参数在 eval 时被扔掉。cosine space 从未被直接优化，只是 ComparisonHead 损失的副产品。这也解释了 seen (0.6109) < unseen (0.6325) 的反常现象。

**修复**：`train_at_v3.py` eval 改为 `logit, _, _, _ = model(e, txts, q)` + `sigmoid(logit)`，与训练目标一致。

### 🔴 P0：AA 的 comparison head 从未被训练（不适用于当前代码）

当前 `train_aa.py` 的 `AudioAudioModel` 没有 comparison head，直接返回 cosine 相似度用于训练和 eval，因此不存在此问题。

### 🟠 P1：Hard Neg 10x 过采样导致记忆

**问题**：`n_hard_ep = min(300000, len(hard_dedup) * 10)` 让同一批难负例在一个 epoch 被看 10 次，模型过拟合到特定词对（如 hi↔haier），而非学到通用区分能力。

**修复**：改为 `* 2` 的轻度过采样，配合每 epoch 重新随机采样，确保不同 epoch 看到不同的 hard neg 组合。

### 🟠 P1：文本 GRU 过深（4 层处理 2-6 音素）

**问题**：`PhonemeBiGRUEncoder` 使用 4 层 BiGRU + dropout=0.1，但大多数英文单词只有 2-6 个音素。第 3-4 层传递已收敛的表示，dropout 可能随机丢掉关键音素。

**修复**：改为 2 层 BiGRU，dropout=0.0。

### 🟡 P2：噪声增强范围过大

**问题**：每步随机 SNR ∈ [-10, 5] dB，-10 dB 让信号被噪声淹没，模型难以学到干净的声学特征。

**修复**：改为分档概率采样 — 70% 轻度 (0~5dB)，20% 中度 (-5~0dB)，10% 重度 (-10~-5dB)。

### 🟡 P2：pos_weight 与数据比例不匹配

**问题**：pos_weight=5.0 + 数据层面 pos:neg=1:5 → 等效 pos:neg=1:1，但 10x 过采样让负样本重复率高。

**修复**：pos_weight 改为 2.0，配合减少过采样，等效 pos 权重略高，适合"宁可误唤醒不可漏唤醒"的 KWS 场景。



## 后续计划（目标 dev AUC 0.9）

由于目前的最佳模型AT V8，对于text  与 音损正常的 query audio之间的判断是较为可信的，然而若音损过度则难以分辨到底是真不一样还是假不一样，需要AA参与辅助判断，如何设计一种更好的模型依赖判断逻辑是值得思考的事情。

- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
- Chung et al., "In Defence of Metric Learning for Speaker Recognition", Interspeech 2020
- Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training", 2021
- Wan et al., "Generalized End-to-End Loss for Speaker Verification", ICASSP 2018
