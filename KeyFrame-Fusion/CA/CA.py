import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    KFC-KWS 论文 Section 2.1 公式 (6) 中的 CA 模块：
    用相似度矩阵 M_am 引导全局上下文特征 F_m^c 进行交叉注意力运算
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 针对 Query (M_am) 和 Key/Value (F_m^c) 的线性映射
        # 由于 M_am 维度是 [B, T_pq, T_sm]，我们需要将其投影到 embed_dim 维度
        self.q_proj = nn.LazyLinear(embed_dim) 
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, M_am: torch.Tensor, F_mc: torch.Tensor) -> torch.Tensor:
        """
        参数:
            M_am: 相似度矩阵 [B, T_pq, T_sm] (由 CS 模块输出)
            F_mc: 全局特征   [B, T_sm, D]   (由 QbyOmni 分支输出)
            
        返回:
            H_am: 交叉注意力融合后的特征 [B, T_pq, D]
        """
        # 1. 将 M_am 映射到标准隐藏维度 D -> [B, T_pq, D]
        Q = self.q_proj(M_am)
        
        # 2. 对全局特征做 K, V 投影 -> [B, T_sm, D]
        K = self.k_proj(F_mc)
        V = self.v_proj(F_mc)
        
        # 3. 计算 Multi-Head Cross-Attention
        # Q 关注 K，从 V 中提取融合后的特征
        attn_out, _ = self.multihead_attn(query=Q, key=K, value=V)
        
        # 4. 残差与归一化
        out = self.norm(Q + self.dropout(attn_out))
        return out


class QbyKeyframeCABranch(nn.Module):
    """
    并行封装三路 CA (对应图中的三个 CA 模块)
    对音频-音频、音频-文本、音频-音素三路分别做 Cross-Attention
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.ca_aa = CrossAttentionBlock(embed_dim, num_heads)
        self.ca_at = CrossAttentionBlock(embed_dim, num_heads)
        self.ca_ap = CrossAttentionBlock(embed_dim, num_heads)

    def forward(self, cs_outputs: dict, omni_features: dict) -> dict:
        """
        参数:
            cs_outputs: 包含 M_aa, M_at, M_ap 的字典
            omni_features: 包含 F_a_c, F_t_c, F_p_c 的字典
            
        返回:
            包含 H_aa, H_at, H_ap 的结果字典
        """
        M_aa, M_at, M_ap = cs_outputs["M_aa"], cs_outputs["M_at"], cs_outputs["M_ap"]
        F_a_c, F_t_c, F_p_c = omni_features["F_a_c"], omni_features["F_t_c"], omni_features["F_p_c"]

        # 1. 音频-音频 路 CA
        H_aa = self.ca_aa(M_aa, F_a_c) if M_aa is not None else None
        
        # 2. 音频-文本 路 CA
        H_at = self.ca_at(M_at, F_t_c)
        
        # 3. 音频-音素 路 CA
        H_ap = self.ca_ap(M_ap, F_p_c)

        return {
            "H_aa": H_aa,
            "H_at": H_at,
            "H_ap": H_ap
        }


# ═══════════ 单元运行测试 ═══════════
if __name__ == "__main__":
    B, D = 2, 128
    T_pq = 12  # Query 关键帧数量
    T_sa, T_st, T_sp = 8, 5, 10  # 各种模态的序列长度

    # 1. 模拟 CS 模块输出的相似度矩阵
    cs_outputs = {
        "M_aa": torch.randn(B, T_pq, T_sa),
        "M_at": torch.randn(B, T_pq, T_st),
        "M_ap": torch.randn(B, T_pq, T_sp)
    }

    # 2. 模拟 QbyOmni 输出的全局特征
    omni_features = {
        "F_a_c": torch.randn(B, T_sa, D),
        "F_t_c": torch.randn(B, T_st, D),
        "F_p_c": torch.randn(B, T_sp, D)
    }

    # 3. 运行 CA 模块
    ca_branch = QbyKeyframeCABranch(embed_dim=D, num_heads=4)
    ca_outputs = ca_branch(cs_outputs, omni_features)

    print("=== CA 模块输出特征维度 ===")
    print("H_aa 维度:", ca_outputs["H_aa"].shape)  # [2, 12, 128]
    print("H_at 维度:", ca_outputs["H_at"].shape)  # [2, 12, 128]
    print("H_ap 维度:", ca_outputs["H_ap"].shape)  # [2, 12, 128]