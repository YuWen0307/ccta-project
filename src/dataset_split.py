"""
数据集划分模块：
- 基于患者ID的随机划分(train/val/test)
- 输出划分CSV
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def patient_level_split(patient_ids, cfg):
    """
    患者级划分。
    patient_ids: list[str]
    返回 dict: {split_name: [pid,...]}
    """
    s = cfg.split
    rng = np.random.default_rng(s.random_seed)

    ids = list(patient_ids)
    rng.shuffle(ids) # 随机打乱顺序

    n = len(ids)
    n_train = int(n * s.train_ratio) # 计算训练集大小
    n_val = int(n * s.val_ratio)
    # test 取剩余，保证覆盖
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    # 若test为空但val不为空，从val里匀一个
    if not test_ids and len(val_ids) >= 2:
        test_ids = [val_ids.pop()]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_split_csv(splits, out_path):
    """保存split.csv,每行(patient_id, split)"""
    rows = []
    for split_name, pids in splits.items():
        for pid in pids:
            rows.append({"patient_id": pid, "split": split_name})
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True) # 确保父目录存在
    df.to_csv(out_path, index=False) # 保存到CSV文件
    return df, out_path


if __name__ == "__main__":
    # 测试划分
    fake_pids = [f"patient_{i:03d}" for i in range(20)]
    from .config import load_config
    cfg = load_config()
    splits = patient_level_split(fake_pids, cfg)
    for k, v in splits.items():
        print(k, len(v))