"""
可视化模块：
- CT切片+mask叠加(轴/矢/冠三视图)
- 中心线在MIP上叠加
- 每个患者的QC质控图
- 标签颜色映射
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 在无GUI环境下使用matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # 用于图例中的彩色方块
from pathlib import Path


def _setup_cjk_font():
    """
    为 matplotlib 配置可用的中文字体，避免图内中文渲染成方块。
    按优先级在系统已安装字体中查找，找不到则回退到 matplotlib 默认字体。
    返回实际选中的字体名列表。
    """
    from matplotlib import font_manager

    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
        "Source Han Sans SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [c for c in candidates if c in available]

    if chosen:
        existing = list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["font.sans-serif"] = chosen + [f for f in existing if f not in chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


CJK_FONTS = _setup_cjk_font()  # 配置中文字体，避免图内中文渲染成方块

# 标签颜色（RGB，用于叠加显示）
LABEL_COLORS = {
    1: (255, 0, 0),      # RCA: 红
    2: (255, 165, 0),    # LM: 橙
    3: (0, 255, 0),      # LAD: 绿
    4: (0, 0, 255),      # LCX: 蓝
    5: (255, 255, 0),    # RAMUS: 黄
    6: (255, 0, 255),    # D1: 洋红
    7: (0, 255, 255),    # D2: 青
    8: (255, 255, 255),  # POINT: 白
}
_DEFAULT_FALLBACK_COLOR = (180, 180, 180)  # 未在 LABEL_COLORS 中定义时的默认灰色

def _label_color(lid):
    """安全地获取标签颜色，未定义时回退到灰色。"""
    return LABEL_COLORS.get(lid, _DEFAULT_FALLBACK_COLOR)

def hu_to_rgb(ct_arr, hu_min=-100, hu_max=800):
    """HU值归一化转灰度RGB 避免除零:hu_min == hu_max 时返回全黑图。"""
    denom = float(hu_max - hu_min)
    if denom == 0:
        norm = np.zeros_like(ct_arr, dtype=np.float32)
    else:
        norm = (ct_arr.astype(np.float32) - hu_min) / denom
    norm = np.clip(norm, 0, 1)  # 限制在 [0, 1] 范围内
    # 通道扩展：将单通道归一化灰度值在通道维度复制三次，得到 (H, W, 3) 的三通道灰度图
    # 此时R/G/B三个通道的数值完全相等 视觉上呈现为灰度效果 完成单通道CT数据到灰度RGB格式的转换
    return np.stack([norm, norm, norm], axis=-1)

def overlay_mask_on_slice(ct_slice, mask_slice, alpha=0.4, hu_min=-100, hu_max=800):
    """
    在单张CT切片上叠加彩色mask。
    ct_slice: 2D HU数组
    mask_slice: 2D 标签数组
    alpha: 掩码颜色的透明度权重
    hu_min/hu_max: CT灰度窗宽窗位;与外部窗宽配置一致时使用
    返回 (H,W,3) uint8 图像。
    """
    rgb = hu_to_rgb(ct_slice, hu_min=hu_min, hu_max=hu_max)
    overlay = (rgb * 255).astype(np.uint8)  # astype 已返回新数组，无需再 .copy()

    for lid, color in LABEL_COLORS.items():  # 遍历每个标签类别及其对应颜色
        mask = (mask_slice == lid)  # 创建布尔掩码，标记当前标签在 mask_slice 中的位置
        if not mask.any():
            continue
        color_arr = np.array(color, dtype=np.float32)  # 将颜色元组转换为 ndarray，便于逐通道计算
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                (1 - alpha) * overlay[:, :, c] + alpha * color_arr[c],
                overlay[:, :, c]
            ).astype(np.uint8)  # mask 位置按 alpha 混合，其余位置保持原值
    return overlay

def _axes_at(axes, row, col):
    """
    兼容 axes 可能是 1D(单切片)或 2D 的情况：
    - n_slices_per_view == 1 时 plt.subplots(3, 1) 返回的是 1D axes 数组；
    - 通用地取 (row, col)，避免 IndexError。
    """
    if getattr(axes, "ndim", None) == 2:
        return axes[row, col]
    return axes[row] if col == 0 else axes[col]

def plot_three_view_qc(ct_arr, mask_arr, patient_id, out_path, cfg):
    """
    绘制三视图(轴/矢/冠)的CT+mask叠加,多个切片排列。
    每个方向取n_slices个等间隔切片。
    """
    n = cfg.qc.n_slices_per_view # 每个方向取的切片数量
    alpha = cfg.qc.overlay_alpha # mask叠加的透明度
    hu_min, hu_max = cfg.preprocessing.clip_intensity  # 统一窗宽窗位

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12)) # 创建一个3行n列的子图网格，每个子图用于显示一个切片
    fig.suptitle(f"Patient: {patient_id} — CT + Coronary Mask Overlay", fontsize=14) # 设置整个图的标题，显示患者ID和图像内容

    views = [
        ("Axial", 0),   # z
        ("Sagittal", 2), # x  (矢状面沿x方向)
        ("Coronal", 1),  # y  (冠状面沿y方向)
    ]

    for row, (vname, axis) in enumerate(views):
            size = ct_arr.shape[axis]
            # 跳过边界，取中间 30%~70% 区域的等间隔切片，避开边缘噪声/不完整信息
            idx = np.linspace(size * 0.3, size * 0.7, n).astype(int)
            for col, sl in enumerate(idx):
                ax = _axes_at(axes, row, col) if n > 1 else axes[row]
                if axis == 0:
                    ct_sl = ct_arr[sl, :, :]
                    msk_sl = mask_arr[sl, :, :]
                elif axis == 1:
                    ct_sl = ct_arr[:, sl, :]
                    msk_sl = mask_arr[:, sl, :]
                else:
                    ct_sl = ct_arr[:, :, sl]
                    msk_sl = mask_arr[:, :, sl]
                overlay = overlay_mask_on_slice(ct_sl, msk_sl, alpha=alpha,
                                                hu_min=hu_min, hu_max=hu_max)
                ax.imshow(overlay, origin="lower")  # 统一 origin，与 dashboard 一致
                ax.set_title(f"{vname} slice {sl}", fontsize=9)
                ax.axis("off")
    # 图例
    legend_elems = [
        Patch(facecolor=np.array(c) / 255.0, label=f"{lid}: {name}")
        for lid, c, name in [
            (1, (255, 0, 0), "RCA"),
            (2, (255, 165, 0), "LM"),
            (3, (0, 255, 0), "LAD"),
            (4, (0, 0, 255), "LCX"),
            (5, (255, 255, 0), "RAMUS"),
            (6, (255, 0, 255), "D1"),
            (7, (0, 255, 255), "D2"),
        ]
    ]
    # 创建图例元素列表，每个元素是一个Patch对象，表示不同标签的彩色方块，并附上标签名称
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    # 设置图例，使用Patch对象创建彩色方块表示不同标签，并在图的下方居中显示
    plt.tight_layout(rect=[0, 0.06, 1, 0.96]) # 调整子图布局，确保图例和标题不会被遮挡，rect参数指定了子图区域的边界
    out_path = Path(out_path) # 将输出路径转换为Path对象，方便后续文件操作
    out_path.parent.mkdir(parents=True, exist_ok=True) # 创建输出路径的父目录，如果不存在则创建，确保保存图像时不会报错
    plt.savefig(out_path, dpi=cfg.qc.dpi, bbox_inches="tight")
    # 保存图像到指定路径，使用配置中的DPI设置图像分辨率，并裁剪多余的空白区域
    plt.close(fig) # 关闭图像，释放内存资源，避免在批量处理时占用过多内存
    return out_path # 返回保存图像的路径，方便后续使用或记录

def _pad_to_shape(arr, target_h, target_w):
    """
    用常量黑边把 2D 图居中补齐到 (target_h, target_w)。
    返回 (padded, pad_top, pad_left) pad_top/pad_left 用于同步平移叠加的散点坐标。
    只做边界填充，不缩放、不重采样，因此不会改变解剖比例。
    """
    h, w = arr.shape[:2]
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    if arr.ndim == 3:  # (H, W, C)
        pads = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:              # (H, W)
        pads = ((pad_top, pad_bottom), (pad_left, pad_right))
    return np.pad(arr, pads, mode="constant", constant_values=0), pad_top, pad_left


def plot_mip_with_centerline(ct_arr, mask_arr, skel_dict, patient_id, out_path, cfg):
    """
    画 MIP(最大密度投影)并把各血管中心线点画在上面。

    三视图的 MIP 形状天然不同 Axial=(Y,X)、Coronal=(Z,X)、Sagittal=(Z,Y)。
    imshow 默认 aspect="equal"，会按各自的数据长宽比压缩子图，
    导致三张图渲染尺寸不一致（宽度相同、高度各异）。
    这里采用「补黑边」方案：先把三张 MIP 居中补齐到同一 (H, W)
    再把中心线散点坐标同步平移，使三张图等大且不产生任何形变。

    skel_dict: {label_id: skeleton_3d_bool_array}
    """
    hu_min, hu_max = cfg.preprocessing.clip_intensity
    denom = float(hu_max - hu_min) or 1.0  # 窗宽为 0 时防除零

    # (投影轴, 视图名, 散点水平坐标索引 col, 散点垂直坐标索引 row)
    # axis=0: MIP(Y,X)  col=x->2, row=y->1
    # axis=1: MIP(Z,X)  col=x->2, row=z->0
    # axis=2: MIP(Z,Y)  col=y->1, row=z->0
    views = [
        (0, "Axial MIP", 2, 1),
        (1, "Coronal MIP", 2, 0),
        (2, "Sagittal MIP", 1, 0),
    ]

    # 先投影一次，确定三视图统一画布尺寸（各视图的最大高/宽）
    mips = [np.max(ct_arr, axis=axis) for axis, _, _, _ in views]
    target_h = max(m.shape[0] for m in mips)
    target_w = max(m.shape[1] for m in mips)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Patient: {patient_id} — MIP with Centerlines", fontsize=13)

    for ax, (axis, name, c_idx, r_idx), mip in zip(axes, views, mips):
        mip_disp = np.clip((mip - hu_min) / denom, 0, 1)
        # 补黑边到统一画布，并记录居中偏移量
        mip_padded, pad_top, pad_left = _pad_to_shape(mip_disp, target_h, target_w)
        ax.imshow(mip_padded, cmap="gray", origin="lower")

        # 把所有中心线点投影到该视图；坐标需加上补边偏移以对齐中心
        for lid, skel in skel_dict.items():
            if skel is None:
                continue
            pts = np.argwhere(skel > 0)
            if pts.size == 0:
                continue
            lname = cfg.labels.names.get(lid, f"L{lid}")
            color = np.array(_label_color(lid)) / 255.0
            # origin="lower" 下 scatter 第一参为水平坐标、第二参为垂直坐标；
            # 补边只是整体平移，两轴各加 (pad_left, pad_top) 即可对齐。
            ax.scatter(pts[:, c_idx] + pad_left, pts[:, r_idx] + pad_top,
                       s=0.3, color=color, label=lname)
        ax.set_title(name)
        ax.set_aspect("equal")  # 三视图数据形状已统一，等比例显示即为等大
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=np.array(_label_color(lid)) / 255.0,
                   markersize=6, label=cfg.labels.names.get(lid, f"L{lid}"))
        for lid in skel_dict.keys()
        if skel_dict[lid] is not None and np.count_nonzero(skel_dict[lid]) > 0
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=8, bbox_to_anchor=(0.5, -0.005))
    plt.tight_layout()
    # 顶部下压、底部上收：给总标题与图例分别留出更大的空白。
    # top 越小 → 总标题与子图标题空隙越大；bottom 越大 → 图像与图例空隙越大。
    fig.subplots_adjust(top=0.8628, bottom=0.115)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=cfg.qc.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_qc_dashboard(ct_arr, mask_arr, spacing, skel_dict, vol_df,
                      branch_dict, patient_id, out_path, cfg):
    """
    自动化质量检查 QC 诊断看板(10项检查)
    ┌─────────────────────────────────────────────────────────┐
    │ ① 患者信息卡(间距/shape/方向)     ② 检查结论(PASS/WARN)  │
    │ ③ HU分布直方图(空气/软组织/钙化)  ④ 标签体素数柱状图      │
    │ ⑤ 三视图mask覆盖率热力条          ⑥ 各血管中心线长度条    │
    │ ⑦ 轴位切片抽查(上/中/下三档)      ⑧ 冠状/矢状切片抽查     │
    └─────────────────────────────────────────────────────────┘

    参数说明：
    - vol_df: 预留给后续血管体积 DataFrame 展示，当前版本仅做 QC 检查，
              保留在签名中以不破坏既有调用方接口。
    """
    qc_warnings = []  # 收集告警项（改名避免遮蔽标准库 warnings）

    # ---- 基础诊断 ----
    vox_mm = float(np.prod(spacing))  # 体素体积 (mm³)
    shape = ct_arr.shape  # CT 三维数组原始尺寸

    sp_min = float(min(spacing))
    sp_max = float(max(spacing))
    isotropic_ratio = (sp_max / sp_min) if sp_min > 0 else float("inf")
    if isotropic_ratio > 1.3:
        qc_warnings.append(f"体素非各向同性(ratio={isotropic_ratio:.2f})")

    # mask 总体积占比
    total_mask_vox = int((mask_arr > 0).sum())
    mask_coverage = total_mask_vox / mask_arr.size * 100 if mask_arr.size > 0 else 0.0
    if mask_coverage < 0.05:
        qc_warnings.append(f"mask 体素占比仅 {mask_coverage:.3f}%，可能标注不全")
    if mask_coverage > 10:
        qc_warnings.append(f"mask 占比 {mask_coverage:.1f}% 偏高，检查标签是否外溢")

     # 各标签体素数
    present_labels = []  # 实际存在的标签 [(name, vox_count), ...]
    missing_labels = []  # 完全缺失的标签（不含豁免标签）
    small_labels = []    # 体积极小的标签（不含豁免标签）
    total_labels = len(cfg.labels.names)

    # QC 豁免标签：POINT 为点状标注，体素数量天然极少，
    # 不检查其"是否缺失"，也不要求其最小体积（纳入检查必然误报）。
    # 如需扩展豁免范围，直接往下面两个集合里加即可（按 id 或按标签名，名称不区分大小写）。
    skip_label_ids = {8}          # 对应 LABEL_COLORS 中的 POINT
    skip_label_names = {"POINT"}  # 按标签名豁免（不区分大小写）

    def _is_skipped(lid, lname):
        """命中豁免 id 或豁免名称，则该标签不参与"缺失/极小体积"两类检查。"""
        return lid in skip_label_ids or str(lname).strip().upper() in skip_label_names

    for lid, lname in cfg.labels.names.items():
        n = int((mask_arr == lid).sum())
        skipped = _is_skipped(lid, lname)  # POINT 等标签豁免检查
        if n == 0:
            if not skipped:  # 豁免标签即使缺失也不计入"缺失"告警
                missing_labels.append(lname)
        else:
            present_labels.append((lname, n))
            # 豁免标签不要求最小体积；其余小于 3 mm^3 视为极小
            if not skipped and n * vox_mm < 3:
                small_labels.append(f"{lname}({n*vox_mm:.1f}mm³)")
    if len(missing_labels) > 3:
        qc_warnings.append(f"缺失标签较多: {', '.join(missing_labels)}")
    if small_labels:
        qc_warnings.append(f"极小标签(<3mm³): {', '.join(small_labels)}")

    # 中心线检查：正确统计骨架体素点数（使用 count_nonzero，而非 len(skel_3d_array)）
    if skel_dict:
        skel_total_points = sum(int(np.count_nonzero(pts))
                                for pts in skel_dict.values() if pts is not None)
    else:
        skel_total_points = 0
    if skel_total_points == 0:
        qc_warnings.append("未提取到任何中心线")

    # HU 范围检查
    hu_min_arr, hu_max_arr = float(ct_arr.min()), float(ct_arr.max())
    if hu_min_arr < -2000 or hu_max_arr > 4000:
        qc_warnings.append(f"HU 范围异常 [{hu_min_arr:.0f}, {hu_max_arr:.0f}]")

    # ---- 绘图 ----
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"QC Dashboard — Patient {patient_id}",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.35,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    # 统一窗宽（用于切片叠加灰度归一化，与 MIP 保持一致）
    wl_min, wl_max = cfg.preprocessing.clip_intensity
    overlay_alpha = cfg.qc.overlay_alpha

    # ① 信息卡
    ax_info = fig.add_subplot(gs[0, 0:2])
    ax_info.axis("off")
    info_text = (
        f"Shape: {shape[0]}×{shape[1]}×{shape[2]}\n" # CT影像三维原始尺寸
        f"Spacing(mm): {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f}\n"
        f"Isotropic ratio: {isotropic_ratio:.2f}\n" # 体素各向异性比值
        f"HU range: [{hu_min_arr:.0f}, {hu_max_arr:.0f}]\n" # CT影像HU值的全局范围
        f"Mask coverage: {mask_coverage:.3f}%  ({total_mask_vox} vox)\n" # 掩码区域的体素占比与总有效体素数
        f"Present labels: {len(present_labels)}/{total_labels}\n"
        f"Skeleton points: {skel_total_points}" # 所有类别中心线的总点数
    )
    ax_info.text(0.02, 0.98, info_text, va="top", ha="left", fontsize=10,
                 family="monospace", transform=ax_info.transAxes,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f0f4f8", ec="#88a"))

    # ② 结论卡
    ax_concl = fig.add_subplot(gs[0, 2:4])
    ax_concl.axis("off")
    if not qc_warnings:
        status = "PASS"
        color = "#2e8b57"
        msg = "所有检查项通过"
    elif len(qc_warnings) <= 2:
        status = "WARN"
        color = "#d4880f"
        msg = "\n".join(f"• {w}" for w in qc_warnings)
    else:
        status = "FAIL"
        color = "#c0392b"
        msg = "\n".join(f"• {w}" for w in qc_warnings)
    ax_concl.text(0.5, 0.95, status, va="top", ha="center", fontsize=20,
                  fontweight="bold", color=color, transform=ax_concl.transAxes)
    ax_concl.text(0.5, 0.75, msg, va="top", ha="left", fontsize=9,
                  transform=ax_concl.transAxes,
                  bbox=dict(boxstyle="round,pad=0.5",
                            fc="#fffbe6" if qc_warnings else "#eaffea",
                            ec=color, lw=1.5))

    # ③ HU 分布直方图（纵轴为 voxels 体素数量）
    ax_hu = fig.add_subplot(gs[1, 0:2])
    hu_flat = ct_arr.flatten()
    ax_hu.hist(hu_flat, bins=100, range=(-1024, 1500), color="#5b8db8", edgecolor="none")
    y_top = ax_hu.get_ylim()[1]
    for thr, lbl in [(-1000, "air"), (-100, "fat"), (100, "soft"), (400, "bone")]:
        ax_hu.axvline(thr, color="red", ls="--", lw=0.7, alpha=0.6)
        ax_hu.text(thr + 10, y_top * 0.9, lbl, color="red", fontsize=7)
    ax_hu.set_title("HU distribution", fontsize=10)
    ax_hu.set_xlabel("HU"); ax_hu.set_ylabel("voxels")

    # ④ 各标签体积柱状图
    ax_lbl = fig.add_subplot(gs[1, 2:4])
    if present_labels:
        names = [p[0] for p in present_labels]
        counts = [p[1] * vox_mm for p in present_labels]
        bars = ax_lbl.barh(names, counts, color="#5fba7d")
        ax_lbl.set_xlabel("Volume (mm³)")
        ax_lbl.set_title("Label volumes", fontsize=10)
        max_c = max(counts) if counts else 0
        for b, v in zip(bars, counts):
            ax_lbl.text(v + max_c * 0.01, b.get_y() + b.get_height() / 2,
                        f"{v:.1f}", va="center", fontsize=8)
    else:
        ax_lbl.text(0.5, 0.5, "No labels found!", ha="center", va="center",
                    color="red", fontsize=12, transform=ax_lbl.transAxes)

    # ⑤ 三视图 mask 覆盖率热力条（沿 z/y/x 的每片覆盖率）
    ax_cov = fig.add_subplot(gs[2, 0:2])
    sz0, sy0, sx0 = mask_arr.shape
    area_zy = sy0 * sx0 if sy0 * sx0 > 0 else 1
    area_zx = sz0 * sx0 if sz0 * sx0 > 0 else 1
    area_yx = sz0 * sy0 if sz0 * sy0 > 0 else 1
    z_cov = (mask_arr > 0).sum(axis=(1, 2)) / area_zy * 100
    y_cov = (mask_arr > 0).sum(axis=(0, 2)) / area_zx * 100
    x_cov = (mask_arr > 0).sum(axis=(0, 1)) / area_yx * 100
    ax_cov.plot(z_cov, label="axial (z)", color="#e74c3c", alpha=0.7)
    ax_cov.plot(y_cov, label="coronal (y)", color="#3498db", alpha=0.7)
    ax_cov.plot(x_cov, label="sagittal (x)", color="#2ecc71", alpha=0.7)
    ax_cov.set_title("Mask coverage per slice (%)", fontsize=10)
    ax_cov.set_xlabel("slice index"); ax_cov.set_ylabel("coverage %")
    ax_cov.legend(fontsize=8); ax_cov.grid(alpha=0.3)

    # ⑥ 各血管中心线长度
    ax_cl = fig.add_subplot(gs[2, 2:4])
    cl_names, cl_lens = [], []
    if branch_dict:
        for lid, branches in branch_dict.items():
            lname = cfg.labels.names.get(lid, f"L{lid}")
            tot = sum(b["length_mm"] for b in branches)
            if tot > 0:
                cl_names.append(lname); cl_lens.append(tot)
    if cl_names:
        bars = ax_cl.barh(cl_names, cl_lens, color="#9b59b6")
        max_l = max(cl_lens) if cl_lens else 0
        for b, v in zip(bars, cl_lens):
            ax_cl.text(v + max_l * 0.01, b.get_y() + b.get_height() / 2,
                       f"{v:.1f}mm", va="center", fontsize=8)
    else:
        ax_cl.text(0.5, 0.5, "No centerline lengths", ha="center", va="center",
                   fontsize=10, color="gray", transform=ax_cl.transAxes)
    ax_cl.set_title("Centerline length per vessel", fontsize=10)
    ax_cl.set_xlabel("length (mm)")

    # ⑦⑧ 最后一行用嵌套 subgridspec 1×6，避免轴位/冠状/矢状子图在同一网格格重叠：
    # 第 1 格：标题占位；第 2~4 格：轴位上/中/下三档抽查；
    # 第 5 格：冠状位抽查；第 6 格：矢状位抽查。
    sub = gs[3, :].subgridspec(1, 6, wspace=0.25)

    ax_t = fig.add_subplot(sub[0, 0]); ax_t.axis("off")
    ax_t.set_title("Axial spot-check (upper/mid/lower)", fontsize=10)

    izs = [int(shape[0] * 0.25), int(shape[0] * 0.5), int(shape[0] * 0.75)]
    for i, iz in enumerate(izs):
        ax_sub = fig.add_subplot(sub[0, i + 1])
        ax_sub.imshow(overlay_mask_on_slice(ct_arr[iz], mask_arr[iz],
                                            alpha=overlay_alpha,
                                            hu_min=wl_min, hu_max=wl_max),
                      origin="lower")
        ax_sub.set_title(f"z={iz}", fontsize=8)
        ax_sub.axis("off")

    iy_mid = shape[1] // 2
    ax_cor_sub = fig.add_subplot(sub[0, 4])
    ax_cor_sub.imshow(overlay_mask_on_slice(ct_arr[:, iy_mid, :],
                                            mask_arr[:, iy_mid, :],
                                            alpha=overlay_alpha,
                                            hu_min=wl_min, hu_max=wl_max),
                      origin="lower")
    ax_cor_sub.set_title(f"Coronal y={iy_mid}", fontsize=8)
    ax_cor_sub.axis("off")

    ix_mid = shape[2] // 2
    ax_sag_sub = fig.add_subplot(sub[0, 5])
    ax_sag_sub.imshow(overlay_mask_on_slice(ct_arr[:, :, ix_mid],
                                            mask_arr[:, :, ix_mid],
                                            alpha=overlay_alpha,
                                            hu_min=wl_min, hu_max=wl_max),
                      origin="lower")
    ax_sag_sub.set_title(f"Sagittal x={ix_mid}", fontsize=8)
    ax_sag_sub.axis("off")

    out_path = Path(out_path)  # 补 Path 转换，避免字符串传入时 .parent 报错
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=cfg.qc.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, qc_warnings

if __name__ == "__main__":
    print("可视化模块就绪")
