"""
特征计算模块：
- 标签体积统计、灰度统计(HU均值、标准差、min、max)、空间范围(bounding box)
- 中心线几何特征（长度、管径、曲折度）
- 汇总为DataFrame / CSV
"""
import numpy as np # 用于影像数组的数值运算
import pandas as pd # 用于结构化结果的汇总与导出
from pathlib import Path # 用于安全规范地处理文件路径
from .io_utils import sitk_to_numpy # 将SimpleITK读取的医学影像转换为numpy数组格式
from .centerline import (
    get_binary_mask, extract_skeleton, extract_branches,
    estimate_radius, smooth_centerline,
    branch_length, arc_chord_ratio, sum_of_angles, count_inflections
)
# 二值掩膜生成、骨架提取、分支拆分、半径估算、中心线平滑、分支长度计算、弧弦比计算、角度求和、拐点计数

def label_volume_stats(ct_arr, mask_arr, spacing, labels_map):
    """
    统计每个标签的体积、灰度、空间范围。
    返回list of dict,每行一个标签。
    """
    sp_z, sp_y, sp_x = spacing
    voxel_vol = sp_z * sp_y * sp_x  # mm^3 per voxel

    rows = []
    for lid, lname in labels_map.items():
        if lid == 8 and lname == "POINT":
            # POINT是标记点，不按体区域统计
            pts = np.argwhere(mask_arr == lid) # 获取所有标记点的坐标
            if len(pts) == 0: # 没有标记点
                continue
            phys_pts = pts * np.array([sp_z, sp_y, sp_x]) # 将体素坐标转换为物理坐标
            rows.append({
                "label_id": lid,
                "label_name": lname,
                "n_voxels": len(pts), # 标记点数量
                "volume_mm3": 0.0, # 标记点没有体积
                "hu_mean": float(ct_arr[pts[:, 0], pts[:, 1], pts[:, 2]].mean()), # 标记点的平均灰度值
                "hu_std": 0.0,
                "hu_min": 0.0,
                "hu_max": 0.0,
                "bbox_z_mm": (float(phys_pts[:, 0].min()), float(phys_pts[:, 0].max())), # 在z方向上的边界框的最小和最大物理坐标
                "bbox_y_mm": (float(phys_pts[:, 1].min()), float(phys_pts[:, 1].max())),
                "bbox_x_mm": (float(phys_pts[:, 2].min()), float(phys_pts[:, 2].max())),
            })
            continue

        bm = get_binary_mask(mask_arr, lid) # 获取当前标签的二值掩膜
        if not bm.any(): # 如果该标签在mask中不存在，则记录空值
            rows.append({
                "label_id": lid, "label_name": lname,
                "n_voxels": 0, "volume_mm3": 0.0,
                "hu_mean": np.nan, "hu_std": np.nan, "hu_min": np.nan, "hu_max": np.nan, # # 没有体素时，灰度统计设为NaN
                "bbox_z_mm": (0, 0), "bbox_y_mm": (0, 0), "bbox_x_mm": (0, 0),
            })
            continue

        coords = np.argwhere(bm > 0) # 获取所有标记体素的坐标
        vals = ct_arr[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32) # 提取对应的CT值
        phys = coords * np.array([sp_z, sp_y, sp_x]) # 将体素坐标转换为物理坐标

        rows.append({
            "label_id": lid,
            "label_name": lname,
            "n_voxels": int(len(coords)), # 体素数量
            "volume_mm3": float(len(coords) * voxel_vol), # 体积 = 体素数量 * 每个体素的体积
            "hu_mean": float(vals.mean()), # 平均灰度值
            "hu_std": float(vals.std()), # 灰度标准差
            "hu_min": float(vals.min()), # 最小灰度值
            "hu_max": float(vals.max()), # 最大灰度值
            "bbox_z_mm": (float(phys[:, 0].min()), float(phys[:, 0].max())), # 在z方向上的边界框的最小和最大物理坐标
            "bbox_y_mm": (float(phys[:, 1].min()), float(phys[:, 1].max())),
            "bbox_x_mm": (float(phys[:, 2].min()), float(phys[:, 2].max())),
        })
    return rows


def vessel_geometric_features(mask_arr, spacing, vessel_labels, cfg):
    """
    对每个血管标签：提取骨架->分支->计算长度/平均管径/曲折度。
    返回 list of dict(每标签一行,汇总值)以及详细分支列表。
    """
    cl_cfg = cfg.centerline
    feat_cfg = cfg.features
     
    # cl_cfg和feat_cfg分别获取中心线和特征配置
    # rows用于存储每个标签的汇总特征，all_branches用于存储每个分支的详细信息
    rows = []
    all_branches = []

    # 遍历血管标签‌：
    # 对于每个标签，获取其二进制掩码bm。
    # 如果掩码为空，则添加默认值到rows并继续下一个标签
    for lid in vessel_labels:
        bm = get_binary_mask(mask_arr, lid)
        if not bm.any():
            rows.append({
                "label_id": lid,
                "label_name": cfg.labels.names[lid],
                "n_branches": 0,
                "total_length_mm": 0.0,
                "mean_radius_mm": np.nan,
                "max_radius_mm": np.nan,
                "mean_tortuosity_arc_chord": np.nan,
                "total_angle_deg": 0.0,
                "total_inflections": 0,
            })
            continue
        # 提取骨架
        # 使用extract_skeleton函数提取骨架
        # 如果骨架为空，则添加默认值到rows并继续下一个标签
        skel = extract_skeleton(bm, min_length=cl_cfg.min_length)
        if not skel.any():
            rows.append({
                "label_id": lid, "label_name": cfg.labels.names[lid],
                "n_branches": 0, "total_length_mm": 0.0,
                "mean_radius_mm": np.nan, "max_radius_mm": np.nan,
                "mean_tortuosity_arc_chord": np.nan,
                "total_angle_deg": 0.0, "total_inflections": 0,
            })
            continue
        # 半径估计和分支提取‌
        # 使用estimate_radius函数估计半径,extract_branches函数提取分支
        dist, radii = estimate_radius(bm, skel, spacing=spacing)
        branches = extract_branches(skel, min_length=cl_cfg.min_length)
        # 计算每个分支的长度、曲折度、角度和拐点数量，并收集所有分支的半径信息
        total_len = 0.0 # 存储该标签的总长度
        tortuosities = [] # 存储每个分支的曲折度
        total_angle = 0.0 # 存储该标签的总角度
        total_inf = 0 # 存储该标签的总拐点数量
        valid_radii = [] # 存储该标签所有分支的有效半径
        for idx, bpts in enumerate(branches): # 遍历每个分支的骨架点
            if cl_cfg.smooth: # 如果配置要求平滑中心线，则对骨架点进行高斯平滑
                bpts_sm = smooth_centerline(bpts.astype(np.float64), sigma=cl_cfg.smooth_sigma)
            else:
                bpts_sm = bpts.astype(np.float64)
            L = branch_length(bpts_sm, spacing) # 计算分支长度
            total_len += L
            tr = arc_chord_ratio(bpts_sm, spacing) # 计算分支的曲折度
            if np.isfinite(tr): # 如果曲折度是有限值，则添加到tortuosities列表中
                tortuosities.append(tr)
            total_angle += sum_of_angles(bpts_sm, spacing) # 计算分支的总角度
            total_inf += count_inflections(bpts_sm, spacing) # 计算分支的拐点数量
            # 该分支上骨架点的半径
            for (z, y, x) in bpts:
                zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
                if 0 <= zi < bm.shape[0] and 0 <= yi < bm.shape[1] and 0 <= xi < bm.shape[2]: # 骨架点在图像范围内
                    if skel[zi, yi, xi]: # 骨架点在骨架上
                        valid_radii.append(dist[zi, yi, xi]) # 添加该骨架点的半径到valid_radii列表中
            all_branches.append({
                "label_id": lid, "label_name": cfg.labels.names[lid],
                "branch_idx": idx, "length_mm": L, "n_points": len(bpts)
            }) # 将每个分支的详细信息添加到all_branches列表中
        # 计算平均半径、最大半径和平均曲折度 将汇总特征添加到rows
        mean_radius = float(np.mean(valid_radii)) if valid_radii else np.nan
        max_radius = float(np.max(valid_radii)) if valid_radii else np.nan
        mean_tr = float(np.mean(tortuosities)) if tortuosities else np.nan

        rows.append({
            "label_id": lid,
            "label_name": cfg.labels.names[lid],
            "n_branches": len(branches),
            "total_length_mm": float(total_len),
            "mean_radius_mm": mean_radius,
            "max_radius_mm": max_radius,
            "mean_tortuosity_arc_chord": mean_tr,
            "total_angle_deg": float(total_angle),
            "total_inflections": int(total_inf),
        })
    # 返回汇总特征rows，详细分支信息all_branches，以及骨架skel（如果存在）
    return rows, all_branches, skel if 'skel' in dir() else None


def patient_features(patient_id, ct_image, mask_image, cfg):
    """
    对单个患者计算所有特征，返回：
    - vol_df: 标签级体积/灰度/范围统计表
    - geo_df: 标签级几何特征表
    - branch_df: 分支级详细表
    - meta: 信息dict(spacing, shape等)
    """
    ct_arr, meta = sitk_to_numpy(ct_image) # 将CT影像转换为numpy数组，并获取元信息
    mask_arr, _ = sitk_to_numpy(mask_image) # 将mask影像转换为numpy数组
    spacing = meta["spacing"]  # (x,y,z) in SimpleITK
    # 数组shape是(z,y,x)，所以spacing转换为(z,y,x)
    spacing_zyx = (spacing[2], spacing[1], spacing[0])

    # 标签map
    label_map = cfg.labels.names # 获取标签ID到名称的映射字典

    vol_rows = label_volume_stats(ct_arr, mask_arr, spacing_zyx, label_map) # 计算每个标签的体积、灰度和空间范围统计
    geo_rows, branch_list, skel = vessel_geometric_features(
        mask_arr, spacing_zyx, cfg.labels.vessel_labels, cfg
    ) # 计算每个血管标签的几何特征和分支详细信息

    vol_df = pd.DataFrame(vol_rows) # 将体积统计结果转换为DataFrame
    geo_df = pd.DataFrame(geo_rows) # 将几何特征结果转换为DataFrame
    branch_df = pd.DataFrame(branch_list) if branch_list else pd.DataFrame() # 将分支详细信息转换为DataFrame，如果没有分支则创建空DataFrame

    for df in (vol_df, geo_df): # 将患者ID插入到每个DataFrame的第一列
        df.insert(0, "patient_id", patient_id)
    if not branch_df.empty: # 如果分支DataFrame不为空，则将患者ID插入到第一列
        branch_df.insert(0, "patient_id", patient_id)

    meta_info = {
        "shape_zyx": ct_arr.shape,
        "spacing_zyx": spacing_zyx,
        "spacing_xyz": spacing,
    } # 创建元信息字典，包含影像的形状和间距信息 承载了影像的核心几何属性是后续所有定量特征计算的基础依据
    # 返回体积统计表、几何特征表、分支详细表和元信息字典
    return vol_df, geo_df, branch_df, meta_info


if __name__ == "__main__":
    print("特征计算模块就绪")
