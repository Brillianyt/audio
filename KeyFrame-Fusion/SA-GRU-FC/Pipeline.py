import torch
import torch.nn as nn
import torch.nn.functional as F

class QbyOmniBranch(nn.Module):
    """
    QbyOmni 分支实现：
    包含三路独立的 [Concat -> SA -> GRU + FC] 处理管道
    分别应对：
    1. Enroll Audio + Query Audio
    2. Enroll Text  + Query Audio
    3. Enroll Phone + Query Audio
    """
    def __init__(self, dim: int = 128, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.dim = dim

        # ═══════════ 1. 三路 Self-Attention (SA) 模块 ═══════════
        # 对应论文中的 SA Block (通常采用 Transformer Encoder Block)
        self.sa_audio = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4, batch_first=True)
        self.sa_text  = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4, batch_first=True)
        self.sa_phone = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*4, batch_first=True)

        # ═══════════ 2. 三路 GRU + FC 时序提取模块 ═══════════
        self.gru_audio = nn.GRU(input_size=dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc_audio  = nn.Linear(hidden_dim, dim)

        self.gru_text  = nn.GRU(input_size=dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc_text   = nn.Linear(hidden_dim, dim)

        self.gru_phone = nn.GRU(input_size=dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc_phone  = nn.Linear(hidden_dim, dim)

        # ═══════════ 3. 序列级别预测头 (对应图中右侧分类 FC) ═══════════
        # 映射到 1 维 Logit，用于计算帧级/序列级 BCE 损失 (L_text 和 L_phon)
        self.seq_classifier_text = nn.Linear(dim, 1)
        self.seq_classifier_phone = nn.Linear(dim, 1)

    def _process_single_pipeline(self, query_embed, enroll_embed, sa_layer, gru_layer, fc_layer):
        """
        单路通用处理逻辑：Concat -> SA -> GRU -> FC
        """
        # A. 沿时间维度 (dim=1) 拼接：[B, T_q + T_s, D]
        concat_feat = torch.cat([query_embed, enroll_embed], dim=1)

        # B. Self-Attention 提炼交互特征 -> E_bar
        E_bar = sa_layer(concat_feat)

        # C. GRU + FC 提炼时序表示 -> F_c
        gru_out, _ = gru_layer(E_bar)  # [B, T_q + T_s, hidden_dim]
        F_c = fc_layer(gru_out)        # [B, T_q + T_s, dim]

        return E_bar, F_c

    def forward(
        self, 
        E_q_a: torch.Tensor,  # Query Audio 特征   [B, T_q, D]
        E_s_a: torch.Tensor,  # Enroll Audio 特征  [B, T_sa, D]
        E_s_t: torch.Tensor,  # Enroll Text 特征   [B, T_st, D]
        E_s_p: torch.Tensor   # Enroll Phone 特征  [B, T_sp, D]
    ):
        """
        Returns:
            F_a_c, F_t_c, F_p_c: 送往 QbyKeyframe 模块中 CA 的 Key/Value 输入 [B, T, D]
            logits_text, logits_phon: QbyOmni 内部输出的序列概率 Logits [B, T, 1]
        """
        # 1. Pipeline 1: Enroll Audio + Query Audio
        E_bar_a, F_a_c = self._process_single_pipeline(
            E_q_a, E_s_a, self.sa_audio, self.gru_audio, self.fc_audio
        )

        # 2. Pipeline 2: Enroll Text + Query Audio
        E_bar_t, F_t_c = self._process_single_pipeline(
            E_q_a, E_s_t, self.sa_text, self.gru_text, self.fc_text
        )

        # 3. Pipeline 3: Enroll Phone + Query Audio
        E_bar_p, F_p_c = self._process_single_pipeline(
            E_q_a, E_s_p, self.sa_phone, self.gru_phone, self.fc_phone
        )

        # 4. 序列级别预测分支 (如图中右侧分类 FC 所示)
        logits_text = self.seq_classifier_text(F_t_c)   # [B, T_q + T_st, 1]
        logits_phon = self.seq_classifier_phone(F_p_c)  # [B, T_q + T_sp, 1]

        # 返回输出：F_*_c 将送去与 Keyframe 算出的相似度矩阵做 Cross-Attention
        return {
            "F_a_c": F_a_c,
            "F_t_c": F_t_c,
            "F_p_c": F_p_c,
            "logits_text": logits_text,
            "logits_phon": logits_phon
        }


# ═══════════ 单元运行与测试 ═══════════
if __name__ == "__main__":
    B, D = 2, 128
    
    # 模拟上一步编码器输出的 4 个张量
    E_q_a = torch.randn(B, 50, D)   # Query Audio (50帧)
    E_s_a = torch.randn(B, 30, D)   # Enroll Audio (30帧)
    E_s_t = torch.randn(B, 10, D)   # Enroll Text (10词)
    E_s_p = torch.randn(B, 15, D)   # Enroll Phone (15音素)

    # 实例化 QbyOmni 模块
    omni_module = QbyOmniBranch(dim=D)

    # 前向计算
    omni_outputs = omni_module(E_q_a, E_s_a, E_s_t, E_s_p)

    print("=== QbyOmni 模块输出特征维度 ===")
    print("送往 CA 的 F_a_c 维度:", omni_outputs["F_a_c"].shape)  # [2, 80, 128]  (50+30)
    print("送往 CA 的 F_t_c 维度:", omni_outputs["F_t_c"].shape)  # [2, 60, 128]  (50+10)
    print("送往 CA 的 F_p_c 维度:", omni_outputs["F_p_c"].shape)  # [2, 65, 128]  (50+15)
    print("内部 FC 输出 logits_text 维度:", omni_outputs["logits_text"].shape) # [2, 60, 1]
    print("内部 FC 输出 logits_phon 维度:", omni_outputs["logits_phon"].shape) # [2, 65, 1]