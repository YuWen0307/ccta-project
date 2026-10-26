#!/usr/bin/env python
"""
主入口脚本：完整 CT/CCTA 冠脉处理流水线（脚本版）
================================================
流程：
1. 发现 data/raw/ 下所有 (CTA, mask) 成对患者
2. 先做患者级 train/val/test 划分
3. 对每个患者执行：
     DCM(zip)+nii 读取 → HU裁剪/重采样/对齐/ROI裁剪 → 保存预处理
     → 体积/灰度/空间范围统计 → skimage骨架化中心线
     → 管径/长度/曲折度几何特征 → QC可视化(三视图+MIP)
     → 生成该患者 Markdown 报告
4. 所有产物写入 outputs/per_patient/<split>/<patient_id>/

用法：
    python scripts/run_pipeline.py [--config configs/default.yaml]
"""
import sys
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# 把项目根加入 sys.path，保证 src.* 可导入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.io_utils import (
    read_dcm_series_from_zip, read_nifti, save_nifti,
    discover_patients, sitk_to_numpy,
)
from src.preprocessing import preprocess_patient
from src.features import patient_features
from src.centerline import get_binary_mask, extract_skeleton
from src.visualization import (
    plot_three_view_qc, plot_mip_with_centerline, plot_qc_dashboard,
)
from src.dataset_split import patient_level_split, save_split_csv

sys.path.insert(0, str(Path(__file__).parent))
from report import build_patient_report, save_patient_report


# ---------------------------------------------------------------
# 单患者处理
# ---------------------------------------------------------------
def process_one_patient(pid, split_name, cta_path, mask_path, cfg):
    """
    处理单个患者，产物全部落到 outputs/per_patient/<split_name>/<pid>/
    """
    t0 = time.time()
    result = {"patient_id": pid, "split": split_name, "status": "ok", "error": None}

    pdir = Path(cfg.output.per_patient_dir) / split_name / pid
    pdir.mkdir(parents=True, exist_ok=True)

    # 1. 读取
    print(f"[{pid}] 读取CTA: {cta_path.name}")
    cta = read_dcm_series_from_zip(cta_path)
    print(f"[{pid}] 读取mask: {mask_path.name if mask_path else 'None'}")
    if mask_path is None:
        raise FileNotFoundError(f"{pid} 未找到对应mask文件")
    mask = read_nifti(mask_path)

    # 2. 预处理
    print(f"[{pid}] 预处理中...")
    cta_pp, mask_pp, bbox = preprocess_patient(cta, mask, cfg)

    # 3. 保存预处理结果
    if cfg.preprocessing.save_nifti:
        pp_dir = Path(cfg.data.processed_dir) / pid
        pp_dir.mkdir(parents=True, exist_ok=True)
        ct_out = pp_dir / f"{pid}_ct.nii.gz"
        mk_out = pp_dir / f"{pid}_mask.nii.gz"
        save_nifti(cta_pp, ct_out)
        save_nifti(mask_pp, mk_out)
        result["ct_nifti"] = str(ct_out)
        result["mask_nifti"] = str(mk_out)
        result["bbox"] = bbox

    ct_arr, meta = sitk_to_numpy(cta_pp)
    mask_arr, _ = sitk_to_numpy(mask_pp)

    # 4. 特征统计
    print(f"[{pid}] 计算特征...")
    vol_df, geo_df, branch_df, meta_info = patient_features(pid, cta_pp, mask_pp, cfg)

    # 先做骨架提取（给MIP和QC看板共用）
    skel_dict = {}
    branch_dict = {}
    from src.centerline import extract_branches, smooth_centerline, branch_length
    sp_xyz = meta_info["spacing_zyx"][::-1]  # (z,y,x) -> (x,y,z) for centerline funcs
    for lid in cfg.labels.vessel_labels:
        bm = get_binary_mask(mask_arr, lid)
        if bm.any():
            pts = extract_skeleton(bm, min_length=cfg.centerline.min_length)
            skel_dict[lid] = pts
            raw_branches = extract_branches(pts, min_length=max(cfg.centerline.min_length // 2, 3))
            branches_out = []
            for seg in raw_branches:
                smoothed = smooth_centerline(seg, sigma=cfg.centerline.smooth_sigma)
                branches_out.append({
                    "points": smoothed,
                    "length_mm": branch_length(smoothed, spacing=sp_xyz),
                })
            branch_dict[lid] = branches_out
        else:
            skel_dict[lid] = None
            branch_dict[lid] = []

    # 5. QC 可视化 3张：三视图叠加 / MIP+中心线 / QC诊断看板
    print(f"[{pid}] 生成质控图...")
    qc_overlay = pdir / f"{pid}_overlay.png"
    plot_three_view_qc(ct_arr, mask_arr, pid, qc_overlay, cfg)

    qc_mip = pdir / f"{pid}_centerline_mip.png"
    plot_mip_with_centerline(ct_arr, mask_arr, skel_dict, pid, qc_mip, cfg)

    qc_dash = pdir / f"{pid}_qc_dashboard.png"
    qc_dash, warnings = plot_qc_dashboard(
        ct_arr, mask_arr, meta_info["spacing_zyx"][::-1],
        skel_dict, vol_df, branch_dict, pid, qc_dash, cfg
    )

    # 6. 患者级 CSV
    vol_csv = pdir / f"{pid}_volume_stats.csv"
    geo_csv = pdir / f"{pid}_geometric_features.csv"
    branch_csv = pdir / f"{pid}_branches.csv"
    vol_df.to_csv(vol_csv, index=False)
    geo_df.to_csv(geo_csv, index=False)
    if not branch_df.empty:
        branch_df.to_csv(branch_csv, index=False)

    files = {
        "ct_nifti": result.get("ct_nifti"),
        "mask_nifti": result.get("mask_nifti"),
        "qc_overlay": str(qc_overlay),
        "qc_mip": str(qc_mip),
        "qc_dashboard": str(qc_dash),
        "vol_csv": str(vol_csv),
        "geo_csv": str(geo_csv),
        "branch_csv": str(branch_csv) if not branch_df.empty else None,
    }

    # 7. Markdown 报告（带告警信息）
    report_text = build_patient_report(
        pid, split_name, vol_df, geo_df, branch_df, meta_info, files, cfg,
        warnings=warnings,
    )
    report_path = save_patient_report(report_text, pdir / f"{pid}_report.md")

    result.update({
        "report": str(report_path),
        "qc_overlay": str(qc_overlay),
        "qc_mip": str(qc_mip),
        "qc_dashboard": str(qc_dash),
        "vol_csv": str(vol_csv),
        "geo_csv": str(geo_csv),
        "warnings": warnings,
        "elapsed_sec": round(time.time() - t0, 2),
        "shape": meta_info["shape_zyx"],
        "spacing": meta_info["spacing_zyx"],
    })
    return result


# ---------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------
def run_pipeline(config_path=None):
    cfg = load_config(config_path)
    print("=" * 60)
    print("CT/CCTA 冠脉分割处理流水线（逐患者报告版，脚本式）")
    print("=" * 60)

    patients = discover_patients(
        cfg.data.raw_dir,
        cta_pattern=cfg.data.cta_pattern,
        mask_suffix=cfg.data.mask_suffix,
    )
    if not patients:
        print(f"[警告] 在 {cfg.data.raw_dir} 未发现任何CTA数据。")
        print("请将 *_cta.zip 和 *_mask.nii.gz 放入 data/raw/ 目录后重新运行。")
        _write_demo_info(cfg)
        return None

    print(f"共发现 {len(patients)} 位患者数据。")

    # 先划分
    all_pids = list(patients.keys())
    splits = patient_level_split(all_pids, cfg)
    pid_to_split = {pid: s for s, ids in splits.items() for pid in ids}
    print("数据集划分:")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} 例")

    split_df, split_csv = save_split_csv(splits, Path(cfg.output.reports_dir) / "split.csv")
    print(f"划分清单已保存: {split_csv}")

    results, failed = [], []
    for pid, paths in tqdm(patients.items(), desc="Processing patients"):
        split_name = pid_to_split.get(pid, "unknown")
        try:
            res = process_one_patient(pid, split_name, paths["cta"], paths["mask"], cfg)
            results.append(res)
        except Exception as e:
            traceback.print_exc()
            failed.append({"patient_id": pid, "error": str(e)})
            print(f"[错误] {pid}: {e}")

    print("\n" + "=" * 60)
    print("处理完成。")
    print(f"逐患者报告目录: {cfg.output.per_patient_dir}")
    print("目录结构: outputs/per_patient/<split>/<patient_id>/")
    print("  每个患者文件夹内含:")
    print("    - <pid>_report.md              患者级报告")
    print("    - <pid>_qc_dashboard.png       自动化 QC 诊断看板")
    print("    - <pid>_overlay.png            CT+mask 三视图叠加")
    print("    - <pid>_centerline_mip.png     MIP + 中心线投影")
    print("    - <pid>_volume_stats.csv       体积/灰度/空间范围")
    print("    - <pid>_geometric_features.csv 管径/长度/曲折度")
    print("    - <pid>_branches.csv           分支级明细")
    print(f"预处理中间产物: {cfg.data.processed_dir}")
    print(f"数据划分清单:   {Path(cfg.output.reports_dir) / 'split.csv'}")
    print("=" * 60)

    if failed:
        fail_path = Path(cfg.output.reports_dir) / "failed_patients.csv"
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        print(f"[警告] {len(failed)} 例处理失败，详见 {fail_path}")

    return {"results": results, "failed": failed, "splits": splits}


def _write_demo_info(cfg):
    demo_note = Path(cfg.data.raw_dir) / "README.txt"
    demo_note.write_text(
        "请把CTA和标注数据放到本目录(data/raw/)，命名格式：\n"
        "  患者ID_cta.zip       —— zip压缩包,内含该患者的DCM序列文件\n"
        "  患者ID_mask.nii.gz   —— 对应的冠脉血管标注NIfTI文件\n"
        "\n标签约定:\n"
        "  1=RCA 2=LM 3=LAD 4=LCX 5=RAMUS 6=D1 7=D2 8=POINT\n"
        "\n放置完成后运行:python scripts/run_pipeline.py\n",
        encoding="utf-8"
    )
    print(f"数据放置说明已写入: {demo_note}")


def main():
    parser = argparse.ArgumentParser(description="CT/CCTA 冠脉处理流水线")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径，默认使用 configs/default.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
