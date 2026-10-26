# CT/CCTA 冠脉血管分割处理项目

一套完整、可运行的冠状动脉 CTA 影像处理 Python 管线
覆盖**数据集划分→数据读取→预处理→中心线提取→影像/几何特征→可视化→QC质控图**全流程。

## 目录结构

```
project/
├── configs/
│   └── default.yaml                        # 全流程配置（分辨率、标签、划分比例等）
├── data/
│   ├── raw/                                # 原始数据（ZIP+nii.gz 放这里）
│   └── processed/                          # 预处理后输出（重采样+裁剪后的NIfTI）
├── src/
│   ├── __init__.py
│   ├── config.py                           # 配置加载
│   ├── dataset_split.py                    # 患者级 train/val/test 划分
│   ├── io_utils.py                         # DCM(zip)/NIfTI 读写
│   ├── preprocessing.py                    # HU裁剪/各向同性重采样/对齐/ROI裁剪
│   ├── centerline.py                       # 3D骨架化 + 分支分解 + 中心线几何（skimage）
│   ├── features.py                         # 体积/灰度/空间范围/管径/长度/曲折度
│   └── visualization.py                    # 三视图叠加 + MIP中心线 + QC诊断看板
├── scripts/
│   ├── run_pipeline.py                     # 主流水线入口（读取→预处理→可视化→中心线→特征→QC→报告）
│   ├── report.py                           # 逐患者 Markdown 报告渲染
├── outputs/
│   ├── reports/                            # 存放 split.csv（数据划分清单）+ failed_patients.csv
│   └── per_patient/<split>/<patient_id>/   # 逐患者结果（Markdown报告+QC看板+叠加图+MIP+CSV表）
├── tests/
│   └── test_pipeline.py                    # 单元测试
├── .gitignore                              # git配置
├── environment.yml                         # conda 环境
└── README.md
```

## 标签约定

| ID | 名称 | 含义 |
|----|------|------|
| 1 | RCA  | 右冠状动脉 |
| 2 | LM   | 左主干 |
| 3 | LAD  | 左前降支 |
| 4 | LCX  | 左回旋支 |
| 5 | RAMUS| 中间支 |
| 6 | D1   | 第一对角支 |
| 7 | D2   | 第二对角支 |
| 8 | POINT| 标记点（单独处理） |

## 开始

### 1. 搭建环境

```bash
cd project
conda env create -f environment.yml
conda activate ccta-pipeline
```

### 2. 运行单元测试

```bash
python tests/test_pipeline.py
```

### 3. 放置真实数据

将你的真实数据按以下命名放到 `data/raw/`：
```
data/raw/
├── patient001_cta.zip        # zip内为DCM序列文件（支持多级目录）
├── patient001_mask.nii.gz    # 冠脉标注，标签值1~8
├── patient002_cta.zip
├── patient002_mask.nii.gz
...
```

> 标注文件若不带 `_mask` 后缀、直接叫 `patient001.nii.gz` 也能被自动识别。

### 4. 运行管线

```bash
python scripts/run_pipeline.py
```

也可指定自定义配置：
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

## 功能模块说明

### 数据集划分 (`src/dataset_split.py`)
- **患者级随机划分**（无数据泄露）：默认 train 70% / val 15% / test 15%；
- 固定 `random_seed=42` 保证可复现；
- 输出 `outputs/reports/split.csv`。

### 数据读取 (`src/io_utils.py`)
- `read_dcm_series_from_zip()`：直接从zip压缩包读取DCM序列，自动解包并按几何顺序排序，无需手动解压。
- `read_nifti()`：读取nii.gz标注。
- `discover_patients()`：扫描raw目录自动发现(CTA, mask)成对患者。

### 预处理 (`src/preprocessing.py`)
1. **HU值裁剪**：默认 `[-100, 800]` HU，去除空气与骨/金属伪影；
2. **各向同性重采样**：默认 `0.5×0.5×0.5 mm³`，CT用BSpline插值，mask用最近邻避免标签模糊；
3. **CT/mask对齐**：mask自动重采样到CT网格，保证体素对应；
4. **ROI裁剪**：根据mask的bounding box裁剪出心脏区域，pad=5体素，显著减小体积。

### 中心线提取 (`src/centerline.py`)
- 基于 `skimage.morphology.skeletonize_3d` 的3D细化；
- 通过**26邻域端点/交点检测**分解骨架为独立分支；
- 高斯平滑中心线点序列；
- **距离变换估计半径**：在血管二值区域做EDT，骨架点处的距离值即为半径估计。

### 特征计算 (`src/features.py`)

**标签级统计**（每个血管/标记点一行）：
- `n_voxels` / `volume_mm3` ：体素数与体积；
- `hu_mean / hu_std / hu_min / hu_max` ：血管腔内HU灰度统计；
- `bbox_z/y/x_mm` ：空间bounding box范围。

**几何特征**（每个血管标签一行）：
- `n_branches`：分支数；
- `total_length_mm`：中心线总长度（逐点累加欧氏距离）；
- `mean_radius_mm / max_radius_mm`：平均/最大管径半径；
- `mean_tortuosity_arc_chord`：平均弧弦比（弧长/弦长，越大越弯）；
- `total_angle_deg`：相邻切向量偏转角之和；
- `total_inflections`：曲率方向变化（拐点）次数。

同时会输出**分支级详细表** `*_branches.csv`，每分支一行长度和点数。

### 可视化 (`src/visualization.py`)
- **三视图叠加 (`overlay_mask_on_slice`)**：轴/矢/冠三个方向各取 `qc.n_slices_per_view` 张等间隔切片（默认 5 张），CT 灰度图上按固定颜色叠加各血管标签（透明度 `qc.overlay_alpha`，默认 0.4）；
- **MIP 中心线投影 (`plot_centerline_mip`)**：三个方向的最大密度投影，叠加各血管骨架点，直观核对中心线是否落在血管腔内；
- **自动化 QC 诊断看板 (`plot_qc_dashboard`)**：4×2 多面板图（详见下方"自动化 QC 诊断看板"节），自动检测数据质量问题并输出 PASS / WARN 结论；
- **标签颜色固定**：RCA=红 / LM=橙 / LAD=绿 / LCX=蓝 / RAMUS=黄 / D1=洋红 / D2=青；
- **字体处理**：matplotlib 绘图统一采用系统可用的中文字体（优先 Noto Sans CJK、WenQuanYi、SimHei、PingFang、Heiti SC，最后回退 DejaVu Sans），避免 Linux/Windows/Mac 平台下中文方块问题；非 ASCII 标签缺失时会自动打印警告而非直接崩溃；
- **每个患者的 QC 结果**连同报告一起，放在 `outputs/per_patient/<split>/<patient_id>/` 独立文件夹。

## 输出结果

### 数据集划分先于处理
管线**先完成患者级 train/val/test 划分**，再逐患者处理

### 划分清单与失败记录 (`outputs/reports/`)
```
split.csv                      # train/val/test划分清单（唯一保留的全局文件）
failed_patients.csv            # 处理失败的病例（如有，仅用于排查）
```

### 预处理NIfTI (`data/processed/<pid>/`)
```
<pid>_ct.nii.gz                # 重采样+裁剪后的CT（HU值已裁剪）
<pid>_mask.nii.gz              # 对齐后的整数标签mask
```

### 逐患者结果 (`outputs/per_patient/<split>/<patient_id>/`)
```
<pid>_report.md                # 患者级 Markdown 报告（含QC结论、划分归属、体积/灰度/空间范围、几何特征、分支明细、质控文件清单）
<pid>_qc_dashboard.png         # 自动化QC诊断看板（4×2多面板+自动告警，详见下节）
<pid>_overlay.png              # 三视图CT+彩色mask叠加（轴/矢/冠各n_slices_per_view张）
<pid>_centerline_mip.png       # 三方向MIP+彩色中心线投影
<pid>_volume_stats.csv         # 该患者各标签体积/灰度/bbox
<pid>_geometric_features.csv   # 该患者各血管长度/管径/曲折度
<pid>_branches.csv             # 分支级长度明细
```
其中 `<split>` 为 `train` / `val` / `test`，`<pid>` 为患者 ID。

### 自动化 QC 诊断看板

`*_qc_dashboard.png` 是每例患者一张的"体检报告"，用于快速排查数据/标注/预处理问题，4×2 面板布局如下：

| 面板位置 | 内容 | 作用 |
|---------|------|------|
| 左上 ① | 信息卡（shape / spacing / HU 范围 / mask 覆盖率 / 标签数 / 骨架点数） | 基础数据一览 |
| 右上 ② | 自动诊断结论（PASS / WARN 列表） | ⚠ 一眼看到问题 |
| 中左 ③ | HU 直方图（标注 air/fat/soft/bone 分界） | 判断 HU 裁剪是否合理、是否存在异常值 |
| 中右 ④ | 各标签体素数柱状图 | 排查漏标/极小标签 |
| 下左 ⑤ | 三视图（z/y/x）切片覆盖率折线 | 排查 mask 在某方向断层/错位 |
| 下右 ⑥ | 各血管中心线长度柱状图 | 判断血管是否断裂/异常短 |
| 底部行 ⑦⑧ | 轴位 25%/50% 切片、冠状正中、矢状正中彩叠图 | 直观核对 mask 对齐与位置 |

**自动检测的 8 项质控告警**（任何一项命中会在面板②与报告开头列出）：

1. `isotropic_ratio` 超标：重采样后 z/y/x 间距比 > 1.3（体素各向同性失败）；
2. `mask_coverage` 过低/过高：mask 占 CT 比例 < 0.05%（标注过少）或 > 10%（外溢）；
3. `label_missing`：8 个标签中缺失 > 3 个；
4. `tiny_label`：任一标签体积 < 3 mm³（疑似噪声点）；
5. `skeleton_empty`：骨架化后无中心线点（mask 非管状或预处理失败）；
6. `hu_range_abnormal`：HU 超出 [-2000, 4000]（DICOM 解析异常）；
7. `slice_coverage_gap`：切片覆盖率曲线有断层（可视化辅助判断，不直接报错）；
8. 底部切片抽查：四方向关键切片彩叠，人工最后核对对齐情况。

## 关键参数调整

编辑 `configs/default.yaml`：
- `preprocessing.target_spacing`：重采样目标间距（显存/精度权衡），如 `[0.4, 0.4, 0.4]` 更精细，`[0.7, 0.7, 0.7]` 更省内存；
- `preprocessing.clip_intensity`：HU裁剪窗口，常规冠脉 CTA 推荐 [-100, 800]；
- `preprocessing.crop_pad`：ROI 裁剪在 mask bbox 外多保留的体素数（默认 5），过小可能截断血管边缘；
- `centerline.smooth_sigma`：中心线高斯平滑强度（体素单位），越大中心线越光滑但可能丢失细节；
- `centerline.min_branch_length`：最短分支保留阈值（mm），过滤毛刺小分支；
- `split.train_ratio / val_ratio / test_ratio`：划分比例（默认 0.7/0.15/0.15，三者之和必须为 1）；
- `split.random_seed`：划分随机种子（默认 42，保证可复现）；
- `qc.n_slices_per_view`：QC 叠加图每方向等间隔切片数（默认 5，值越大图片越高但更慢）；
- `qc.overlay_alpha`：mask 叠加透明度（0~1，默认 0.4，越小越能看到下层 CT 细节）；
- `qc.dpi`：QC 图保存 DPI（默认 150，越高越清晰但文件越大）；
- `qc.save_per_patient`：是否将 QC 结果保存到逐患者文件夹（默认 true，建议保持开启）。

## 依赖说明

本项目坚持**纯 Python 可安装**原则，不依赖 VMTK/ITK 外部C++扩展库，核心依赖：
- **SimpleITK**：DICOM/NIfTI读写、重采样
- **scikit-image**：3D骨架化（替代VMTK）
- **scipy**：距离变换、卷积
- **numpy/pandas**：数值计算与表格
- **matplotlib/seaborn**：可视化
- **nibabel**：备选NIfTI读取

## 扩展建议

- 中心线若需要更专业的结果，可在 `src/centerline.py` 中切换为 VMTK 的 `vmtkcenterlines`（需要额外安装）；
- 如需训练深度学习分割模型，可基于 `data/processed/` 中重采样后的NIfTI直接接入 MONAI / nnU-Net ；
- 逐患者特征表 (`*_geometric_features.csv`) 可在后续脚本中自行汇总后用于狭窄检测、斑块分类等下游统计或机器学习任务；
- QC 告警项可按需在 `src/visualization.py::plot_qc_dashboard` 中增加自定义规则（如 HU 均值过低、曲折度异常等）。
