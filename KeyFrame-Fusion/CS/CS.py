import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityMatrix(nn.Module):
    """
    KFC-KWS 论文公式 (5) 实现：
    计算 Query 音频关键帧特征与注册特征之间的余弦相似度矩阵 M_am
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, E_q_key: torch.Tensor, E_s_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_q_key: Query 音频关键帧向量  [B, T_pq, D]
            E_s_feat: Enroll 模态特征向量   [B, T_sm, D] (可以是 E_s_a_key, E_s_t, 或 E_s_p)
            
        Returns:
            M_am: 相似度矩阵             [B, T_pq, T_sm]
        """
        # 1. 沿通道维度 (dim=-1) 进行 L2 范数归一化，防止 0 除
        E_q_norm = F.normalize(E_q_key, p=2, dim=-1, eps=self.eps)  # [B, T_pq, D]
        E_s_norm = F.normalize(E_s_feat, p=2, dim=-1, eps=self.eps) # [B, T_sm, D]

        # 2. 批量矩阵乘法算点积 (相当于对每个向量算余弦相似度)[cite: 2]
        # [B, T_pq, D] @ [B, D, T_sm] -> [B, T_pq, T_sm]
        M_am = torch.bmm(E_q_norm, E_s_norm.transpose(1, 2))

        return M_am


class QbyKeyframeCSBranch(nn.Module):
    """
    三路 Cosine Similarity 并行封装模块
    实现图中三个 CS 盒子的功能[cite: 2]
    """
    def __init__(self):
        super().__init__()
        self.cs_calc = CosineSimilarityMatrix()

    def forward(
        self, 
        E_a_q_key: torch.Tensor,  # Query Audio 关键帧   [B, T_pq, D][cite: 2]
        E_s_a_key: torch.Tensor,  # Enroll Audio 关键帧  [B, T_pa, D][cite: 2]
        E_s_t: torch.Tensor,      # Enroll Text 特征      [B, T_st, D][cite: 2]
        E_s_p: torch.Tensor       # Enroll Phone 特征     [B, T_sp, D][cite: 2]
    ):
        """
        Returns:
            M_aa: 音频-音频相似度矩阵 [B, T_pq, T_pa][cite: 2]
            M_at: 音频-文本相似度矩阵 [B, T_pq, T_st][cite: 2]
            M_ap: 音频-音素相似度矩阵 [B, T_pq, T_sp][cite: 2]
        """
        # 1. 计算 Audio-Audio 关键帧相似度[cite: 2]
        M_aa = self.cs_calc(E_a_q_key, E_s_a_key) if E_s_a_key is not None else None

        # 2. 计算 Audio-Text 关键帧相似度[cite: 2]
        M_at = self.cs_calc(E_a_q_key, E_s_t)

        # 3. 计算 Audio-Phone 关键帧相似度[cite: 2]
        M_ap = self.cs_calc(E_a_q_key, E_s_p)

        return {
            "M_aa": M_aa,
            "M_at": M_at,
            "M_ap": M_ap
        }


# ═══════════ 单元运行与测试 ═══════════
if __name__ == "__main__":
    B, D = 2, 128

    # 模拟输入张量
    E_a_q_key = torch.randn(B, 12, D)  # Query Audio 选出的 12 个关键帧[cite: 2]
    E_s_a_key = torch.randn(B, 8, D)   # Enroll Audio 选出的 8 个关键帧[cite: 2]
    E_s_t     = torch.randn(B, 5, D)   # Enroll Text (5 个词)[cite: 2]
    E_s_p     = torch.randn(B, 10, D)  # Enroll Phone (10 个音素)[cite: 2]

    # 实例化 CS 模块
    cs_module = QbyKeyframeCSBranch()

    # 计算相似度矩阵
    cs_outputs = cs_module(E_a_q_key, E_s_a_key, E_s_t, E_s_p)

    print("=== CS 模块输出相似度矩阵维度 ===")
    print("M_aa (Audio-Audio) 维度:", cs_outputs["M_aa"].shape) # [2, 12, 8][cite: 2]
    print("M_at (Audio-Text)  维度:", cs_outputs["M_at"].shape) # [2, 12, 5][cite: 2]
    print("M_ap (Audio-Phone) 维度:", cs_outputs["M_ap"].shape) # [2, 12, 10][cite: 2]