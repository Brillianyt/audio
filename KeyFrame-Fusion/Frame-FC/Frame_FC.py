import torch
import torch.nn as nn

class QbyKeyframeClassifier(nn.Module):
    """
    QbyKeyframe 分支最后的 FC 判别头
    输入: 交叉注意力融合后的关键帧特征 (H_aa, H_at, H_ap)
    输出: 整句级匹配概率 (0 或 1) 与 Loss L_utt
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        # 逐层缩减特征维度
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, H_aa: torch.Tensor, H_at: torch.Tensor, H_ap: torch.Tensor, labels: torch.Tensor = None):
        """
        参数:
            H_aa, H_at, H_ap: 形状均为 [B, T_pq, D] 的特征矩阵
            labels: [B, 1] 真实整句标签 (0/1)[cite: 3]
        """
        # 1. 融合成统一特征 F_k (如论文 Section 2.3 描述的求和融合)[cite: 3]
        valid_feats = [h for h in [H_aa, H_at, H_ap] if h is not None]
        F_k = sum(valid_feats)  # [B, T_pq, D]

        # 2. 沿着关键帧时间维度做平均池化，提炼整句向量
        utt_vector = F_k.mean(dim=1)  # [B, D]

        # 3. 通过 FC 网络映射成 Logit
        logits = self.fc(utt_vector)  # [B, 1]
        probs = torch.sigmoid(logits)

        # 4. 计算 BCE Loss (对应公式 7 L_u)[cite: 3]
        loss = None
        if labels is not None:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels.float())

        return probs, loss


# ---------------- 单元测试 ----------------
if __name__ == "__main__":
    B, T_pq, D = 4, 12, 128
    H_aa = torch.randn(B, T_pq, D)
    H_at = torch.randn(B, T_pq, D)
    H_ap = torch.randn(B, T_pq, D)
    labels = torch.randint(0, 2, (B, 1)).float()

    model = QbyKeyframeClassifier(embed_dim=D)
    probs, loss = model(H_aa, H_at, H_ap, labels)
    print("QbyKeyframe 判别概率维度:", probs.shape)  # [4, 1]
    print("QbyKeyframe Utt Loss:", loss.item() if loss is not None else "N/A")