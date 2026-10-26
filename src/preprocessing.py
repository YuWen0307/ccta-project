"""
预处理模块：
- HU值裁剪
- 各向同性重采样
- CT与mask对齐处理
- 基于mask的ROI裁剪

原因：
-原始 CCTA 的 HU 值范围极宽,而血管的 HU 大概在 -100~800 区间
-原始扫描层厚通常不均匀,‌各向异性会让管径和曲折度算错‌，必须重采样成各向同性
-整幅 CT 里心脏只占很小一块,‌先裁剪 ROI 能把体积降一个量级‌,训练和分析都更快
"""
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from .io_utils import sitk_to_numpy, numpy_to_sitk, save_nifti
    
# HU值裁剪
def clip_hu(image, hu_min=-100, hu_max=800):
    """裁剪HU值范围"""
    arr, meta = sitk_to_numpy(image) # SimpleITK图像 -> numpy数组 + 仿射矩阵 + 头信息
    arr = np.clip(arr, hu_min, hu_max).astype(np.int16) # 裁剪HU值范围
    return numpy_to_sitk(arr, meta) # numpy数组 + 仿射矩阵 + 头信息 -> SimpleITK图像

# ct图像重采样
def resample_isotropic(image, target_spacing=(0.5, 0.5, 0.5), is_mask=False): 
    """ct重采样到各向同性分辨率"""
    original_spacing = image.GetSpacing() 
    original_size = image.GetSize()

    # 新尺寸(体素个数) = 原物理尺寸 ÷ 目标间距‌,物理覆盖范围不变 体素数按目标分辨率重算
    new_size = [
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / target_spacing[2])),
    ]

    resampler = sitk.ResampleImageFilter() # SimpleITK重采样滤波器
    resampler.SetOutputSpacing(target_spacing) # 设置输出间距
    resampler.SetSize(new_size) # 设置输出尺寸
    resampler.SetOutputDirection(image.GetDirection()) # 设置输出方向
    resampler.SetOutputOrigin(image.GetOrigin()) # 设置输出原点
    resampler.SetTransform(sitk.Transform()) # 设置输出变换为空变换（不改变位置）

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # mask用最近邻插值，避免产生非整数标签
        resampler.SetDefaultPixelValue(0) # mask的默认值为0,边界兜底
    else:
        resampler.SetInterpolator(sitk.sitkBSpline) # CT用B样条插值，平滑连续
        resampler.SetDefaultPixelValue(-100)  # 空气HU,边界兜底

    resampled = resampler.Execute(image) # 执行重采样
    return resampled # 返回重采样后的SimpleITK图像

# mask按之前ct图像重采样的网格对齐，保证两者体素数、spacing、origin 完全一致
def resample_to_reference(moving, reference, is_mask=False):
    # 调用函数时is_mask=True,覆盖了is_mask=False
    """将moving图像对齐到reference的网格(CT已经重采样完成时,把mask对齐过去)"""
    resampler = sitk.ResampleImageFilter() # 创建重采样滤波器
    resampler.SetReferenceImage(reference) # 设置参考图像
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # mask用最近邻插值，避免产生非整数标签
        resampler.SetDefaultPixelValue(0) # mask的默认值为0,边界兜底
    else:
        resampler.SetInterpolator(sitk.sitkLinear) # CT用线性插值，平滑连续
        resampler.SetDefaultPixelValue(-100) # 空气HU,边界兜底
    return resampler.Execute(moving) # 执行重采样，返回对齐后的SimpleITK图像

# 基于mask的ROI裁剪
def crop_to_mask(ct_image, mask_image, pad=5):
    """
    用 mask 的非零体素求出包围盒bounding box
    基于mask的bounding box裁剪CT和mask,保留心脏区域,降低体积。
    返回裁剪后的图像以及裁剪bbox信息。
    """
    mask_arr, mask_meta = sitk_to_numpy(mask_image)
    ct_arr, ct_meta = sitk_to_numpy(ct_image)

    # 找所有非零体素
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        return ct_image, mask_image, None # 空mask容错兜底

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    # 加pad并clip边界
    s_z, s_y, s_x = mask_arr.shape
    z_min = max(z_min - pad, 0)
    y_min = max(y_min - pad, 0)
    x_min = max(x_min - pad, 0)
    z_max = min(z_max + pad + 1, s_z)
    y_max = min(y_max + pad + 1, s_y)
    x_max = min(x_max + pad + 1, s_x)

    ct_cropped = ct_arr[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_cropped = mask_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    # 更新origin（裁剪后origin需平移）
    sp = mask_meta["spacing"]
    new_origin = (
        mask_meta["origin"][0] + x_min * sp[0] * mask_meta["direction"][0],
        mask_meta["origin"][1] + y_min * sp[1] * mask_meta["direction"][4],
        mask_meta["origin"][2] + z_min * sp[2] * mask_meta["direction"][8],
    )
    # 每一个轴的新origin坐标 = 原origin对应轴坐标 + 该轴切掉的体素数 × 该轴体素间距 × 该轴方向符号
    # 方向矩阵是标准单位阵(d00=d11=d22=1,其余为0，99%常规冠脉CT都是这个情况)
    new_meta = dict(mask_meta) # 复制原mask的meta信息
    new_meta["origin"] = new_origin # 更新origin

    # 把前面裁剪好的numpy数值数组，和更新过origin的空间元信息重新打包成标准SimpleITK图像对象
    # ct_cropped：前面用numpy切片得到的、裁剪到血管ROI范围内的HU值三维数组
    # mask_cropped：前面用numpy切片得到的、裁剪到血管ROI范围内的mask三维数组
    # new_meta：用前序步骤计算好的、已经更新过裁剪后新origin的元信息
    ct_out = numpy_to_sitk(ct_cropped.astype(np.int16), new_meta)
    mask_out = numpy_to_sitk(mask_cropped.astype(np.uint8), new_meta)

    bbox = {
        "z_min": int(z_min), "z_max": int(z_max),
        "y_min": int(y_min), "y_max": int(y_max),
        "x_min": int(x_min), "x_max": int(x_max),
    }
    return ct_out, mask_out, bbox

# HU值归一化
def normalize_intensity(arr, hu_min=-100, hu_max=800):
    """将HU值归一化到0~1"""
    out = (arr.astype(np.float32) - hu_min) / (hu_max - hu_min)
    return np.clip(out, 0.0, 1.0) # 鲁棒性兜底：强制截断到合法区间


def preprocess_patient(cta_image, mask_image, cfg):
    """
    单个患者完整预处理流程：
    1. HU裁剪
    2. 重采样到各向同性
    3. mask对齐到CT
    4. 基于mask的ROI裁剪
    返回处理后的ct, mask, bbox
    """
    pp = cfg.preprocessing # 预处理参数配置

    # 1. HU裁剪
    cta_image = clip_hu(cta_image, pp.clip_intensity[0], pp.clip_intensity[1])

    # 2. CT重采样
    cta_resampled = resample_isotropic(
        cta_image, tuple(pp.target_spacing), is_mask=False
    )

    # 3. mask重采样对齐到CT
    mask_resampled = resample_to_reference(
        mask_image, cta_resampled, is_mask=True 
    )

    # 4. 裁剪到ROI
    cta_crop, mask_crop, bbox = crop_to_mask(
        cta_resampled, mask_resampled, pad=pp.crop_pad
    )

    return cta_crop, mask_crop, bbox

if __name__ == "__main__":
    print("预处理模块就绪")
