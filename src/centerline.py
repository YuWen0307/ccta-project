"""
中心线提取模块(基于skimage骨架化)
- 3D骨架化
- 分支分解（连通域+交点分析）
- 中心线平滑
- 几何属性计算：长度、管径、曲折度
"""
import numpy as np
from scipy import ndimage as ndi # 提供后续分支分解、几何属性计算需要的连通域分析、空间测量等工具
# scikit-image库的跨版本兼容
try:
    from skimage.morphology import skeletonize_3d as _skeletonize_3d
except ImportError:
    # 新版scikit-image: skeletonize_3d 改名为 skeletonize(..., method='lee')
    from skimage.morphology import skeletonize as _skeletonize_fn
    def _skeletonize_3d(img):
        return _skeletonize_fn(img, method='lee')
    # Lee算法是专门针对3D二值图像优化的骨架化算法，能够完整保留三维结构的拓扑连通性
from skimage.morphology import remove_small_objects # 用于骨架化后清理图像里的微小噪点生成的无效小连通区域,剔除骨架上的毛刺
from skimage.filters import gaussian # 用于后续对提取出的中心线做高斯平滑

def get_binary_mask(mask_arr, label_id):
    """提取指定标签的二值mask"""
    return (mask_arr == label_id).astype(np.uint8)

def extract_skeleton(binary_mask, min_length=5):
    """
    对二值血管mask进行3D骨架化,返回骨架布尔数组
    使用skimage.morphology.skeletonize_3d (lee方法)
    """
    cleaned = binary_mask.astype(bool).copy() # 将输入二值掩膜转为布尔类型并拷贝，避免修改原始数据
    if min_length > 1:
        # 当参数min_length > 1时，执行连通域过滤逻辑
        # 连通域面积过滤：只保留体素数>=min_length的连通域
        from skimage.measure import label # label函数用于标记连通区域
        lbl = label(cleaned, connectivity=3)
        # 对3D布尔二值数组cleaned执行连通区域标记，输出和输入尺寸完全相同的整数标记数组lbl
        # connectivity=3在3维图像场景下,代表使用26邻接连通判定
        # 后续会基于这个标记结果统计每个连通域的体素总量
        areas = np.bincount(lbl.ravel()) # 统计每一个连通域对应的体素总数
        for lid in range(1, lbl.max() + 1): # 遍历所有连通域标签（从1开始，0是背景）
            if areas[lid] < min_length:
                cleaned[lbl == lid] = False
    skel = _skeletonize_3d(cleaned) # 对清理后的二值mask执行3D骨架化，返回骨架布尔数组
    return skel.astype(bool) # 将骨架布尔数组转换为布尔类型并返回

def smooth_centerline(points, sigma=1.0):
    """
    对中心线点序列做高斯平滑。
    points: Nx3数组,按顺序排列的空间点 (z, y, x)
    返回平滑后的Nx3数组。
    """
    if len(points) < 3 or sigma <= 0:
        return points.astype(np.float64)
    smoothed = np.zeros_like(points, dtype=np.float64)
    for axis in range(3):
        smoothed[:, axis] = gaussian(points[:, axis].astype(np.float64), sigma=sigma, mode="nearest")
        # 对当前遍历的轴,提取该维度下所有点的坐标值,先转换为64位浮点数,再调用一维高斯滤波函数对该维度的坐标序列做平滑;
        # 滤波时使用传入的sigma作为标准差，边界处理模式设为nearest(用边界最近的像素值填充边缘区域,避免边缘出现异常畸变)
        # 最后将平滑后的该维度结果赋值给结果数组的对应列
    return smoothed

def find_endpoints_and_junctions(skel):
    """
    在3D骨架中找端点和交点。
    端点:邻域内恰好有1个骨架点;
    交点:邻域内有3个及以上骨架点
    邻域取26-连通。
    """
    # 定义3x3x3的卷积核,用于计算每个骨架点的邻居数
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    # 每个骨架点的邻居数（不包括自身）
    neighbor_count = ndi.convolve(skel.astype(np.uint8), kernel) - skel.astype(np.uint8)

    endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))
    junctions = np.argwhere((skel > 0) & (neighbor_count >= 3))
    # 端点坐标数组,交点坐标数组,全图每个体素的邻域骨架点计数结果 支持后续拓扑分析
    return endpoints, junctions, neighbor_count

def extract_branches(skel, min_length=5):
    """
    从3D骨架中分解出独立分支(段）。
    返回分支列表,每个分支是Nx3的坐标数组。
    方法：
      1. 从端点出发沿骨架做DFS,遇到交点或另一个端点停止;
      2. 如果没有端点（环形），任取一点开始。
    """
    skel_bool = skel.astype(bool).copy()
    if not skel_bool.any():
        return []
    # 得到所有端点坐标 所有交点坐标 丢弃不需要的邻域计数返回值
    endpoints, junctions, _ = find_endpoints_and_junctions(skel_bool)

    # 将交点标记为不可作为普通路径点通过的位置（DFS遇到即终止）
    junction_set = set(map(tuple, junctions))
    # 初始化访问标记数组,用于记录哪些骨架点已经被访问过,避免重复遍历
    visited = np.zeros_like(skel_bool, dtype=bool)
    branches = []

    def neighbors(pt):
        z, y, x = pt
        out = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < skel_bool.shape[0] and 0 <= ny < skel_bool.shape[1] and 0 <= nx < skel_bool.shape[2]:
                        if skel_bool[nz, ny, nx] and not visited[nz, ny, nx]:
                            out.append((nz, ny, nx))
        return out

    # 优先从端点出发
    starts = [tuple(p) for p in endpoints]
    # 若没有端点，就从所有未访问的骨架点出发
    if not starts:
        starts = [tuple(p) for p in np.argwhere(skel_bool)]

    for start in starts:
        if visited[start]:
            continue
        # BFS/DFS 追踪一条分支
        path = [start]
        visited[start] = True
        cur = start
        # 选下一个点（未访问邻居）
        while True:
            nbs = [n for n in neighbors(cur) if tuple(n) not in junction_set or n == cur]
            # 更严格：找未访问的骨架点
            nbs = [n for n in neighbors(cur)]
            # 排除交点（除非分支起点就在交点上且只有这一步）
            next_candidates = []
            for n in nbs:
                if tuple(n) in junction_set:
                    # 走到交点就停，但把交点加入path作为终点
                    path.append(n)
                    visited[n] = True
                    next_candidates = []
                    break
                next_candidates.append(n)
            if not next_candidates:
                break
            # 走第一个候选（简单实现，对主要管状结构可用）
            cur = next_candidates[0]
            path.append(cur)
            visited[cur] = True
            if tuple(cur) in junction_set:
                break

        if len(path) >= min_length:
            branches.append(np.array(path))

    # 处理剩余未访问的骨架段（交点之间的段等）
    remaining = np.argwhere(skel_bool & ~visited)
    if remaining.size > 0:
        from skimage.measure import label
        labels = label(skel_bool & ~visited, connectivity=3)
        for lid in range(1, labels.max() + 1):
            seg_pts = np.argwhere(labels == lid)
            if len(seg_pts) >= min_length:
                branches.append(seg_pts)

    return branches

def estimate_radius(binary_mask, skeleton, spacing=(1.0, 1.0, 1.0)):
    # binary_mask：血管区域的二值掩膜数组，血管区域标记为1，背景为0
    # skeleton：从血管掩膜中提取得到的单像素宽3D骨架数组
    # spacing：三维图像的体素物理间距，默认值为(1.0, 1.0, 1.0)，单位为毫米
    """
    基于距离变换估计骨架点处的血管半径(mm)。
    对血管区域做距离变换，每个骨架点取距离变换值即近似半径。
    返回与骨架同shape的半径数组(mm)以及所有骨架点的平均/最大半径。
    """
    # 使用scipy.ndimage.distance_transform_edt计算距离变换
    from scipy.ndimage import distance_transform_edt
    # distance_transform_edt 按体素距离计算，再乘以spacing得到物理半径
    dist = distance_transform_edt(binary_mask.astype(bool), sampling=spacing)
    # 通过骨架图的非零位置做索引 从全图距离图中提取所有骨架点对应的距离值 这些值就是每个骨架点对应的血管半径
    radii = dist[skeleton > 0]
    # 全图的距离变换数组 仅包含骨架点半径的一维数组
    return dist, radii

def branch_length(points, spacing=(0.5, 0.5, 0.5)):
    """
    计算分支的物理长度(mm)。
    points: Nx3 (z, y, x)
    """
    if len(points) < 2:
        return 0.0
    sp_z, sp_y, sp_x = spacing
    # 体素坐标转真实物理坐标
    phys = points * np.array([sp_z, sp_y, sp_x])
    # 欧氏距离累加:先求每个相邻点之间的欧氏距离
    diffs = np.diff(phys, axis=0)
    # 最后把所有分段的距离求和 得到整条分支路径的总长度
    length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    return float(length)

def arc_chord_ratio(points, spacing=(0.5, 0.5, 0.5)):
    """
    弧弦比(曲折度指标1): 弧长 / 首尾直线距离。
    越接近1越直;越大越曲折。
    """
    # 计算分支的物理长度
    L = branch_length(points, spacing)
    if len(points) < 2:
        return 1.0
    # 计算首尾点的物理坐标
    sp = np.array(spacing)
    p0 = points[0] * sp
    p1 = points[-1] * sp
    # 计算首尾点之间的欧氏距离(弦长)
    chord = float(np.linalg.norm(p1 - p0))
    if chord < 1e-6:
        return float("inf") # 返回正无穷大浮点值‌ 代表这条路径的曲折度极高
    return L / chord


def sum_of_angles(points, spacing=(0.5, 0.5, 0.5)):
    """
    角度和曲折度(指标2):相邻切向量夹角(度)之和。
    """
    if len(points) < 3:
        return 0.0
    # 体素坐标转真实物理坐标
    sp = np.array(spacing)
    phys = points * sp # 真实物理坐标phys
    vecs = np.diff(phys, axis=0) # 每一段相邻点构成的路径向量vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) # 计算每个路径向量的欧氏范数(长度) norms
    norms = np.clip(norms, 1e-8, None) # 避免除以零的情况，将范数限制在最小值1e-8以上
    vecs = vecs / norms
    # 相邻切向量夹角
    cos_a = np.einsum("ij,ij->i", vecs[:-1], vecs[1:]) # 计算相邻向量的点积，得到夹角余弦值
    cos_a = np.clip(cos_a, -1.0, 1.0) # 将余弦值限制在[-1, 1]范围内，避免数值误差导致的无效输入
    angles = np.arccos(cos_a)  # 计算夹角的弧度值
    return float(np.sum(np.degrees(angles))) # 将弧度转换为角度并求和，得到所有相邻切向量夹角的总和，作为曲折度指标


def count_inflections(points, spacing=(0.5, 0.5, 0.5)):
    """
    曲折度指标3:曲率方向变化(拐点)次数。
    简化：对累积距离-曲率符号变化计数。
    """
    if len(points) < 5:
        return 0
    sp = np.array(spacing)
    phys = points * sp
    # 用二阶差分近似曲率
    d1 = np.diff(phys, axis=0) # 一阶差分，得到每个点的切向量
    d2 = np.diff(d1, axis=0) # 二阶差分，得到每个点的曲率向量
    # 计算曲率符号变化
    sign = np.sign(np.sum(d2[:-1] * d2[1:], axis=1))
    # 符号从+变-或相反算一个拐点
    inflections = int(np.sum(sign[:-1] * sign[1:] < 0))
    return inflections


if __name__ == "__main__":
    print("中心线提取模块就绪 (skimage骨架化)")
