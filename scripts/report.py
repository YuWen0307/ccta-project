"""
逐患者报告模块：
- 把单例患者的特征表（体积/灰度/空间范围、几何特征、分支明细）渲染成 Markdown 报告
- 报告头部包含患者所属数据集划分(train/val/test)
"""
import numpy as np
import pandas as pd
from pathlib import Path


def _fmt(v, nd=3):
    """数值格式化,NaN 显示为 '-'"""
    if v is None:
        return "-"
    if isinstance(v, float) and not np.isfinite(v):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _bbox_str(row):
    """把 z/y/x 三轴范围拼成一行可读文本"""
    try:
        z0, z1 = row["bbox_z_mm"]
        y0, y1 = row["bbox_y_mm"]
        x0, x1 = row["bbox_x_mm"]
        return f"z[{z0:.1f},{z1:.1f}] y[{y0:.1f},{y1:.1f}] x[{x0:.1f},{x1:.1f}] mm"
    except Exception:
        return "-"


def _df_to_markdown(df, columns, headers, nd=3):
    """轻量 Markdown 表格渲染(不依赖 tabulate)"""
    if df is None or df.empty:
        return "_(无数据)_\n"
    cols = [c for c in columns if c in df.columns]
    head = "| " + " | ".join(headers[:len(cols)]) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], nd) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_patient_report(patient_id, split_name, vol_df, geo_df, branch_df,
                         meta_info, files, cfg, warnings=None):
    """
    生成单个患者的 Markdown 报告文本。

    patient_id : str
    split_name : str  train/val/test
    vol_df     : 标签体积/灰度/空间范围表
    geo_df     : 标签几何特征表（长度/管径/曲折度）
    branch_df  : 分支级明细表
    meta_info  : dict,含 shape_zyx / spacing_zyx
    files      : dict,记录该患者产出的QC图等文件路径
    warnings   : list[str],QC 自动诊断出的告警项
    """
    lines = []
    lines.append(f"# 患者报告：{patient_id}\n")
    lines.append(f"**所属数据集划分：`{split_name}`**\n")

    # ---- QC 自动诊断结果 ----
    lines.append("## 0. 自动化质量检查 (QC)\n")
    if not warnings:
        lines.append("> ✅ **PASS** — 自动化质控检查全部通过\n")
    else:
        lines.append("> ⚠ **检测到 {} 个告警项**:\n".format(len(warnings)))
        for w in warnings:
            lines.append(f"> - {w}")
        lines.append("")
    if files.get("qc_dashboard"):
        lines.append(f"![QC Dashboard]({files['qc_dashboard']})\n")

    # ---- 基础信息 ----
    lines.append("## 1. 基础信息\n")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 患者ID | {patient_id} |")
    lines.append(f"| 数据划分 | {split_name} |")
    shp = meta_info.get("shape_zyx")
    if shp is not None:
        lines.append(f"| 体数据尺寸 (z,y,x) | {tuple(int(s) for s in shp)} |")
    sp = meta_info.get("spacing_zyx")
    if sp is not None:
        lines.append(f"| 体素间距 (z,y,x) mm | ({sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f}) |")
    if files.get("ct_nifti"):
        lines.append(f"| 预处理CT | {files['ct_nifti']} |")
    if files.get("mask_nifti"):
        lines.append(f"| 预处理mask | {files['mask_nifti']} |")
    lines.append("")

    # ---- 体积 / 灰度 / 空间范围 ----
    lines.append("## 2. 各标签体积、灰度与空间范围\n")
    lines.append("灰度单位为 HU;空间范围为该标签在物理坐标系下的包围盒(mm)。\n")
    lines.append(_df_to_markdown(
        vol_df,
        columns=["label_id", "label_name", "n_voxels", "volume_mm3",
                 "hu_mean", "hu_std", "hu_min", "hu_max", "bbox"],
        headers=["ID", "名称", "体素数", "体积(mm³)", "HU均值", "HU标准差",
                 "HU最小", "HU最大", "空间范围"],
    ))
    # bbox 单独展开一行文本更易读
    if vol_df is not None and not vol_df.empty:
        lines.append("\n**空间范围明细：**\n")
        for _, row in vol_df.iterrows():
            lines.append(f"- {int(row['label_id'])} {row['label_name']}: {_bbox_str(row)}")
        lines.append("")

    # ---- 几何特征 ----
    lines.append("## 3. 血管几何特征（管径 / 长度 / 曲折度）\n")
    lines.append(_df_to_markdown(
        geo_df,
        columns=["label_id", "label_name", "n_branches", "total_length_mm",
                 "mean_radius_mm", "max_radius_mm",
                 "mean_tortuosity_arc_chord", "total_angle_deg", "total_inflections"],
        headers=["ID", "名称", "分支数", "总长度(mm)", "平均管径(mm)", "最大管径(mm)",
                 "曲折度(弧弦比)", "角度累积(°)", "拐点数"],
    ))

    # ---- 分支明细 ----
    lines.append("\n## 4. 分支级明细\n")
    lines.append(_df_to_markdown(
        branch_df,
        columns=["label_id", "label_name", "branch_idx", "length_mm", "n_points"],
        headers=["ID", "名称", "分支序号", "长度(mm)", "骨架点数"],
    ))

    # ---- 质控文件 ----
    lines.append("\n## 5. 质控与可视化文件\n")
    if files.get("qc_dashboard"):
        lines.append(f"- **QC 自动诊断看板**:`{files['qc_dashboard']}`")
    if files.get("qc_overlay"):
        lines.append(f"- CT+mask三视图叠加:`{files['qc_overlay']}`")
    if files.get("qc_mip"):
        lines.append(f"- MIP中心线投影:`{files['qc_mip']}`")
    if files.get("vol_csv"):
        lines.append(f"- 体积统计表:`{files['vol_csv']}`")
    if files.get("geo_csv"):
        lines.append(f"- 几何特征表:`{files['geo_csv']}`")
    if files.get("branch_csv"):
        lines.append(f"- 分支明细表:`{files['branch_csv']}`")
    lines.append("")

    return "\n".join(lines)


def save_patient_report(text, out_path):
    """写出患者报告 Markdown"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
