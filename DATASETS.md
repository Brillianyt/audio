# 数据集说明

## 赛题提供数据

### 训练集
- **train/train_label.csv**: 50 万对训练样本，8357 个唯一词
- **train/wav.zip**: 训练音频，16kHz 单声道 WAV
- **dev/**: 开发集 1 万对 (seen + unseen)，有标签
- **eval/**: 测试集 10 万对 (seen + unseen)，无标签

### 噪声集
- **datasets/noise/**: 赛题提供的噪声音频
- **datasets/wham/**: WHAM 噪声数据集
- **musan_data/musan/**: MUSAN 噪声数据集

## 外部数据集

### Speech Commands v2
- 来源: Google Speech Commands Dataset v2
- 下载: `wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
- 用途: 提供 32 个原词表没有的新词，扩充词表覆盖
- 处理脚本: `baseline/gen_pairs_external.py`

### LibriSpeech
- 来源: OpenSLR (openslr.org/12)
- 下载: `wget https://www.openslr.org/resources/12/train-clean-100.tar.gz`
- 用途: 扩充正样本音频变体，增加每个词的不同发音人覆盖
- 处理脚本: `baseline/gen_librispeech_pairs.py`

### CMU Pronouncing Dictionary
- 来源: cmudict (通过 pip install cmudict)
- 用途: 英文单词→ARPAbet 音素序列转换，用于 PhonemeBiGRU 文本编码器

## 自生成数据

| 文件 | 说明 |
|------|------|
| `train/self_paired.json` | 同词不同音频配对 (26 万对) |
| `train/self_paired_xl.json` | 扩展自配对 (124 万对) |
| `train/fill_pos_pairs.json` | 补充正样本覆盖全部词 |
| `train/mega_pairs.json` | 混合配对数据集 |
| `train/speech_commands_pairs.json` | Speech Commands 配对 |
| `train/librispeech_pairs.json` | LibriSpeech 配对 |
| `train/cleaned_pairs.json` | 清洗去重后的合并数据集 |

## 难负例挖掘

| 文件 | 方法 | 说明 |
|------|------|------|
| `baseline/hard_neg_whisper.json` | Whisper Embedding 余弦距离 | 8 万对 |
| `baseline/hard_neg_iter1.json` | 迭代挖掘 v1 | 8 万对 |
| `baseline/hard_neg_iter2.json` | 迭代挖掘 v2 | 8 万对 |
| `baseline/hard_neg_phoneme.json` | CMU 音素编辑距离 | 8 万对 |
| `baseline/hard_neg_atv2.json` | AT v2 模型难负例 | 8 万对 |
| `baseline/hard_neg_at_final.json` | AT 最终模型难负例 | 71K 对 |
| `baseline/hard_neg_aa_final.json` | AA 最终模型难负例 | 36K 对 |

## 数据统计 (去重后)

| 类别 | 数量 |
|------|------|
| 正样本 (去重) | 78,846 对 |
| 难负样本 (去重) | 74,624 对 |
| 易负样本 (去重) | 494,592 对 |
| 训练时每 epoch 采样 | 60K pos + 149K hard_neg |
