import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCSelectorModule(nn.Module):
    """
    CTC Selector 模块（论文 Figure 2 完整实现）：
    1. 输入语音特征 E_a，通过 Linear + Softmax 计算帧级音素后验概率
    2. 寻找概率峰值，过滤 blank 并去重，精准定位音素关键帧时间戳
    3. 围绕每个关键帧，应用 (2w + 1) 上下文窗口融合局部声学特征
    4. 输出 keyframe 特征及 CTC logits（用于计算 L_ctc）[cite: 1]
    """
    def __init__(self, dim: int = 128, num_phonemes: int = 64, context_w: int = 2):
        super().__init__()
        self.dim = dim
        self.num_phonemes = num_phonemes
        self.context_w = context_w  # 上下文窗口大小 w (论文实验中 w=2，即 2w+1=5 帧)[cite: 1]

        # 线性投影层：将特征映射到音素词表空间 (+1 代表 CTC 占位符 blank)[cite: 1]
        self.ctc_linear = nn.Linear(dim, num_phonemes + 1)

    def _select_keyframes_per_sample(self, audio_embed: torch.Tensor, probs: torch.Tensor):
        """
        处理单条语音样本的关键帧挑选逻辑（对应论文 Section 2.1 CTC Selector）
        audio_embed: [T, D]
        probs: [T, num_phonemes + 1]
        """
        T, D = audio_embed.shape
        blank_idx = self.num_phonemes  # 约定最后一个维度为 blank 占位符[cite: 1]

        # 寻找每帧最可能的音素 ID[cite: 1]
        pred_tokens = torch.argmax(probs, dim=-1)

        # 约束规则：过滤 blank，且同一音素只保留首次出现的峰值位置（去重规则）[cite: 1]
        selected_indices = []
        seen_tokens = set()
        
        for t in range(T):
            token = pred_tokens[t].item()
            if token != blank_idx and token not in seen_tokens:
                seen_tokens.add(token)
                selected_indices.append(t)

        # 兜底保障：若未预测出任何音素（极罕见），默认选取中间帧
        if len(selected_indices) == 0:
            selected_indices = [T // 2]

        # 对应公式 (4)：围绕每个峰值时间戳 t_k，聚合 (2w + 1) 窗口内的上下文特征[cite: 1]
        keyframe_features = []
        for t_k in selected_indices:
            start_idx = max(0, t_k - self.context_w)
            end_idx = min(T, t_k + self.context_w + 1)
            # 窗内均值 Pooling
            context_feat = audio_embed[start_idx:end_idx, :].mean(dim=0)
            keyframe_features.append(context_feat)

        # 堆叠为 [T_p, D]
        return torch.stack(keyframe_features, dim=0)

    def forward_single(self, E_a: torch.Tensor):
        """
        单路音频特征处理逻辑
        E_a: [B, T, D]
        """
        B, T, D = E_a.shape
        
        # 1. 计算音素后验概率分布 (公式 3)[cite: 1]
        logits = self.ctc_linear(E_a)                   # [B, T, num_phonemes + 1]
        probs = F.softmax(logits, dim=-1)                # [B, T, num_phonemes + 1]
        log_probs = F.log_softmax(logits, dim=-1)        # 用于计算 CTC 损失[cite: 1]

        # 2. 逐样本提取关键帧特征
        batch_keyframes = []
        max_tp = 0
        
        for b in range(B):
            k_feat = self._select_keyframes_per_sample(E_a[b], probs[b])
            batch_keyframes.append(k_feat)
            if k_feat.shape[0] > max_tp:
                max_tp = k_feat.shape[0]

        # 3. 动态 Padding 拼接成标准 Batch 张量 -> [B, Max_T_p, D]
        padded_keyframes = torch.zeros(B, max_tp, D, device=E_a.device)
        for b in range(B):
            tp = batch_keyframes[b].shape[0]
            padded_keyframes[b, :tp, :] = batch_keyframes[b]

        return padded_keyframes, log_probs

    def forward(self, E_a_q: torch.Tensor, E_s_a: torch.Tensor = None):
        """
        同时接收 Query Audio 与 Enroll Audio 特征[cite: 1]
        Returns:
            E_a_q_key: Query 音频的关键帧向量  [B, T_pq, D][cite: 1]
            E_s_a_key: Enroll 音频的关键帧向量 [B, T_ps, D] (若提供)[cite: 1]
            ctc_log_probs_q: 用于计算 Query CTC Loss[cite: 1]
            ctc_log_probs_s: 用于计算 Enroll CTC Loss[cite: 1]
        """
        # 1. 抽取 Query 音频的关键帧[cite: 1]
        E_a_q_key, ctc_log_probs_q = self.forward_single(E_a_q)

        # 2. 抽取 Enroll 音频的关键帧 (若存在)[cite: 1]
        E_s_a_key, ctc_log_probs_s = None, None
        if E_s_a is not None:
            E_s_a_key, ctc_log_probs_s = self.forward_single(E_s_a)

        return {
            "E_a_q_key": E_a_q_key,
            "E_s_a_key": E_s_a_key,
            "ctc_log_probs_q": ctc_log_probs_q,
            "ctc_log_probs_s": ctc_log_probs_s
        }


# ═══════════ 单元运行与测试 ═══════════
if __name__ == "__main__":
    B, D = 2, 128
    
    # 模拟前面编码器输出的 Query 和 Enroll 音频特征
    E_a_q = torch.randn(B, 100, D)  # Query Audio (100帧)[cite: 1]
    E_s_a = torch.randn(B, 60, D)   # Enroll Audio (60帧)[cite: 1]

    # 实例化 CTC 选择器 (假设音素表大小为 64，窗口 w=2)[cite: 1]
    ctc_selector = CTCSelectorModule(dim=D, num_phonemes=64, context_w=2)

    # 传入两路音频特征前向传播[cite: 1]
    outputs = ctc_selector(E_a_q, E_s_a)

    print("=== CTC Selector 模块输出维度 ===")
    print("Query 抽取出的 Keyframe 维度 E_a_q_key:", outputs["E_a_q_key"].shape) # [2, T_pq, 128][cite: 1]
    print("Enroll 抽取出的 Keyframe 维度 E_s_a_key:", outputs["E_s_a_key"].shape) # [2, T_ps, 128][cite: 1]
    print("Query CTC 预测概率 Logits 维度:", outputs["ctc_log_probs_q"].shape)   # [2, 100, 65][cite: 1]
    print("Enroll CTC 预测概率 Logits 维度:", outputs["ctc_log_probs_s"].shape)   # [2, 60, 65][cite: 1]