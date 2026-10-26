"""
基础单元测试:验证关键模块能正常工作
    python tests/test_pipeline.py
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) # 将项目根目录添加到sys.path中，以便导入src模块

def test_imports():
    """测试所有 src 工具模块能正常导入(pipeline 主逻辑在 scripts/run_pipeline.py)"""
    from src import config, io_utils, preprocessing, centerline, features, visualization, dataset_split
    print("所有 src 模块导入成功")

def test_split():
    """测试数据集划分比例正确"""
    from src.config import load_config
    from src.dataset_split import patient_level_split
    cfg = load_config()
    pids = [f"P{i:03d}" for i in range(20)]
    splits = patient_level_split(pids, cfg)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 20
    print(f"划分: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

def test_resample():
    """测试重采样尺寸和间距正确"""
    import SimpleITK as sitk
    from src.preprocessing import resample_isotropic
    img = sitk.GetImageFromArray(np.zeros((60, 60, 60), dtype=np.int16)) # 创建一个60x60x60的空CT影像
    img.SetSpacing((1.0, 1.0, 1.0)) # 设置原始影像的体素间距为1mm
    out = resample_isotropic(img, (0.5, 0.5, 0.5), is_mask=False) # 将影像重采样为0.5mm的各向同性间距
    assert out.GetSpacing() == (0.5, 0.5, 0.5) # 验证重采样后的间距是否为0.5mm
    assert out.GetSize()[0] == 120 # 验证重采样后的尺寸是否为120x120x120
    print(f"重采样OK: size={out.GetSize()}, spacing={out.GetSpacing()}") # 打印重采样后的尺寸和间距信息    

def test_skeleton():
    """测试骨架化能在简单管状mask上得到中心线"""
    from src.centerline import extract_skeleton, extract_branches, branch_length, arc_chord_ratio
    # 构造一根简单的"管子"
    # 这段代码会构造出一根‌轴线沿z轴方向延伸的直圆柱3D二值掩膜‌
    mask = np.zeros((40, 40, 40), dtype=np.uint8) 
    for i in range(30):
        for a in range(-2, 3):
            for b in range(-2, 3):
                if a*a + b*b <= 4:
                    mask[20+a, 20+b, 5+i] = 1
    skel = extract_skeleton(mask, min_length=5) # 对掩膜执行骨架化，过滤掉长度小于5的无效骨架段，校验骨架总点数大于5
    # 断言的固定格式是 assert 布尔判断表达式, 自定义错误提示字符串
    assert skel.sum() > 5, "骨架化点太少" 
    branches = extract_branches(skel, min_length=5)
    assert len(branches) >= 1, "应至少提取出1条分支"
    L = branch_length(branches[0], spacing=(0.5, 0.5, 0.5)) # 计算分支长度，spacing为体素间距 
    assert L > 0, "长度应>0"
    tr = arc_chord_ratio(branches[0], spacing=(0.5, 0.5, 0.5)) # 计算弧长与弦长比值
    assert tr >= 1, "弧弦比应>=1"
    print(f"骨架点: {skel.sum()}, 分支数: {len(branches)}, 长度: {L:.2f}mm, 弧弦比: {tr:.3f}")

if __name__ == "__main__":
    test_imports()
    test_split()
    test_resample()
    test_skeleton()
    print("\n全部基础测试通过 ✓")