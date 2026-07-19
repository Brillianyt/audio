from __future__ import annotations

import os
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 40          
    max_frames: int = 100      


@dataclass
class TrainConfig:
    embed_dim: int = 128   # 从64提升到128，配合加深的CNN
    batch_size: int = 128
    num_workers: int = 4   # Linux服务器可设更大 (如8)
    epochs: int = 15
    lr: float = 1e-3
    pos_weight: float = 5.0
    train_subset: int = 500000
    seed: int = 42
    log_every: int = 100
    noise_aug: bool = True
    noise_snr_db: tuple = (-10, 5)
    noise_mode: str = "gaussian"
    noise_dir: str = ""
    specaug: bool = True                   # SpecAugment 频率/时间掩码
    specaug_freq_mask: int = 6             # 频率掩码最大宽度
    specaug_time_mask: int = 30            # 时间掩码最大宽度


@dataclass
class Paths:
    root: str = ROOT
    train_zip: str = field(default="")
    train_csv: str = field(default="")
    dev_seen_zip: str = field(default="")
    dev_seen_csv: str = field(default="")
    dev_unseen_zip: str = field(default="")
    dev_unseen_csv: str = field(default="")
    eval_seen_zip: str = field(default="")
    eval_seen_csv: str = field(default="")
    eval_unseen_zip: str = field(default="")
    eval_unseen_csv: str = field(default="")
    ckpt_dir: str = field(default="")

    def __post_init__(self):
        r = self.root

        # 训练数据
        self.train_zip = os.path.join(r, "train_subset", "wav.zip")
        self.train_csv = os.path.join(r, "train_subset", "train_label.csv")

        # Dev 数据：自动检测 dev/dev/ 嵌套（本地解压导致）
        dev_base = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(dev_base, "dev")):
            dev_base = os.path.join(dev_base, "dev")
        self.dev_seen_zip = os.path.join(dev_base, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(dev_base, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(dev_base, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(dev_base, "dev_unseen", "dev_unseen_label.csv")

        # Eval 数据
        self.eval_seen_zip = os.path.join(r, "eval", "eval_seen", "wav.zip")
        self.eval_seen_csv = os.path.join(r, "evalcsv_without_label", "eval_seen_without_label.csv")
        self.eval_unseen_zip = os.path.join(r, "eval", "eval_unseen", "wav.zip")
        self.eval_unseen_csv = os.path.join(r, "evalcsv_without_label", "eval_unseen_without_label.csv")

        self.ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")


PATHS = Paths()
AUDIO = AudioConfig()
TRAIN = TrainConfig()
