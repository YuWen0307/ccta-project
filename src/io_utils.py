"""
数据IO工具模块
- 读取zip内DCM序列(CCTA)
- 读取nii.gz标注
- 保存NIfTI
"""
import os
import zipfile # 内置的zip压缩包操作库
import tempfile # 用于创建临时目录
import shutil # 用于删除临时目录
from pathlib import Path # 用于路径操作
from glob import glob # 文件名通配符匹配 后缀判断
import numpy as np # 用于数组操作
import SimpleITK as sitk # 用于医学图像处理 ‌DICOM序列读取首选
import nibabel as nib # 用于读取NIfTI文件 ‌NIfTI格式读写首选

# 输入‌：冠脉CTA影像的zip压缩包路径
# 输出‌：已经正确排序、带完整空间几何信息的3D SimpleITK图像对象
def read_dcm_series_from_zip(zip_path):
    """
    从zip压缩包中读取DCM序列,返回SimpleITK图像。
    zip内可包含多级目录,会自动检索所有.dcm/.dicom文件并按InstanceNumber排序。
    """
    zip_path = Path(zip_path) # 把输入路径统一转成Path对象，兼容字符串/Path两种入参，跨平台无压力
    tmpdir = tempfile.mkdtemp(prefix="cta_dcm_") # 创建带明确前缀cta_dcm_的系统临时目录
    # 解压阶段：try块保证资源安全
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # 递归查找所有dcm文件
        dcm_candidates = []
        for ext in ("*.dcm", "*.DCM", "*.dicom", "*.DICOM"):
            dcm_candidates.extend(glob(str(Path(tmpdir) / "**" / ext), recursive=True))

        # 如果没找到带后缀的，就把所有非目录文件都列出来（某些序列无后缀）
        if not dcm_candidates:
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    if not f.startswith("."): # 过滤掉.开头的隐藏文件
                        dcm_candidates.append(str(Path(root) / f))

        # 空结果明确报错
        if not dcm_candidates:
            raise FileNotFoundError(f"在 {zip_path} 内未找到DICOM文件")

        # 使用SimpleITK读取序列（自动按几何排序）
        reader = sitk.ImageSeriesReader() # 创建SimpleITK的图像序列读取器
        reader.MetaDataDictionaryArrayUpdateOn() # 读取DICOM元数据挂载到图像对象上,后续可以直接取用
        reader.LoadPrivateTagsOn() # 读取私有标签（某些厂商的InstanceNumber在私有标签里）
        series_IDs = reader.GetGDCMSeriesIDs(tmpdir) # 获取GDCM序列ID列表,按照DICOM标签里的几何位置信息排序

        # 如果GDCM能识别到有效序列,优先用几何位置正确排序后的文件列表
        if series_IDs:
            series_file_names = reader.GetGDCMSeriesFileNames(tmpdir, series_IDs[0])
        else:
            # 回退：按文件名排序
            series_file_names = sorted(dcm_candidates)

        # 把排序好的文件列表喂给读取器,执行读取得到3D SimpleITK图像
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        return image # 返回的对象不止是CT值numpy数组，还自带‌完整空间几何信息‌
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True) # 删除临时目录，释放磁盘空间

# 读取nii.gz文件,返回SimpleITK原生图像对象，‌适合需要做几何操作的场景
def read_nifti(nifti_path):
    return sitk.ReadImage(str(nifti_path))

# 使用nibabel读取nii.gz,返回numpy数组+仿射矩阵+头信息，‌适合数值计算/深度学习场景‌
def read_nifti_nib(nifti_path):
    img = nib.load(str(nifti_path))
    return img.get_fdata().astype(np.int16), img.affine, img.header

# 自动递归创建输出目录的父文件夹,保存SimpleITK图像为nii.gz 返回最终的Path对象
def save_nifti(image, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))
    return out_path

def sitk_to_numpy(image):
    """SimpleITK图像 -> numpy数组 (shape: [z, y, x])"""
    arr = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()   # (x, y, z)
    origin = image.GetOrigin()
    direction = image.GetDirection()
    return arr, {
        "spacing": tuple(float(s) for s in spacing),
        "origin": tuple(float(o) for o in origin),
        "direction": tuple(direction),
    }

def numpy_to_sitk(arr, meta):
    """numpy数组 -> SimpleITK图像"""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(meta["spacing"])
    img.SetOrigin(meta["origin"])
    img.SetDirection(meta["direction"])
    return img

def discover_patients(raw_dir, cta_pattern="*_cta.zip", mask_suffix="_mask.nii.gz"):
    """
    扫描raw_dir,自动发现所有(CTA, mask)成对的患者。
    返回 dict: {patient_id: {"cta": path, "mask": path}}
    """
    raw_dir = Path(raw_dir)
    patients = {}
    cta_files = list(raw_dir.glob(cta_pattern))
    for cta in cta_files:
        pid = cta.name.replace("_cta.zip", "")
        mask = raw_dir / f"{pid}{mask_suffix}"
        if mask.exists():
            patients[pid] = {"cta": cta, "mask": mask}
        else:
            # 尝试不带_mask的变体
            alt_mask = raw_dir / f"{pid}.nii.gz"
            if alt_mask.exists():
                patients[pid] = {"cta": cta, "mask": alt_mask}
            else:
                patients[pid] = {"cta": cta, "mask": None}
    return patients


if __name__ == "__main__":
    print("IO工具模块就绪")
    print("支持读取ZIP内DCM序列 + nii.gz标注")
