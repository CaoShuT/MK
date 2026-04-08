# MK - 红外图像大气传输效应蒙特卡洛仿真系统

## 项目简介

本项目是一个基于蒙特卡洛方法的**红外图像大气传输效应仿真系统**，用于模拟云雾天气下大气对红外辐射的散射与吸收效应，生成大气点扩散函数（PSF），并将其与原始红外场景图像进行卷积，实现逼真的大气退化仿真。

主要特性：
- 基于权重修正法（Implicit Capture）的高效光子追踪
- Henyey-Greenstein 相函数向量化散射采样
- 透过率分解（弹道 + 散射分量）
- 支持 MODTRAN5 输出文件（.tp7/.plt）辅助参数优化
- 完整的红外图像退化模型（卷积 + 路径辐射）

---

## 安装方法

```bash
pip install -r requirements.txt
```

依赖库：`numpy >= 1.21.0`，`scipy >= 1.7.0`，`matplotlib >= 3.4.0`，`pandas >= 1.3.0`

---

## 快速开始

### 模式A：纯蒙特卡洛演示（无需 MODTRAN5 文件）

```bash
# 中雾，10 μm，路径 1 km，100万光子
python main.py --mode mc --fog medium --wavelength 10.0 --distance 1.0 --n_photons 1000000

# 浓雾，8 μm，路径 2 km
python main.py --mode mc --fog thick --wavelength 8.0 --distance 2.0 --n_photons 2000000

# 薄雾，512×512 场景图像
python main.py --mode mc --fog thin --scene_size 512
```

### 模式B：MODTRAN5 辅助优化仿真

```bash
# 基本用法
python main.py --mode modtran --tp7 file.tp7 --plt file.plt --wavelength 10.0 --distance 1.0

# 指定场景图像（.npy 格式）
python main.py --mode modtran --tp7 file.tp7 --plt file.plt --image scene.npy

# 高精度仿真
python main.py --mode modtran --tp7 file.tp7 --wavelength 10.0 --distance 1.0 --n_photons 2000000
```

### 模式C：已有图像与 PSF 直接卷积退化

无需蒙特卡洛仿真，直接将已有图像文件和 PSF 文件进行卷积退化：

```bash
# 最简用法（T_total=1.0, L_path=0.0，输出自动命名为 image_degraded.npy）
python main.py --mode convolve --image image.npy --psf psf.npy

# 指定透过率和路径辐射，输出为 .npy
python main.py --mode convolve --image image.png --psf psf.npy --t_total 0.75 --l_path 0.02 --output result.npy

# 输出为图像格式（值域自动归一化至 0~255）
python main.py --mode convolve --image image.png --psf psf.npy --output result.png
```

**支持的输入格式：**

| 参数 | 支持格式 |
|------|----------|
| `--image` | `.npy`、`.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff` |
| `--psf` | `.npy`、`.png`、`.jpg`、`.jpeg`、`.tif`、`.tiff` |
| `--output` | `.npy`（原始 float64）或 `.png`/`.jpg`/`.tif`（归一化 uint8）|

**说明：**
- PSF 会自动做非负截断（负值置零）并重新归一化（积分 = 1）
- RGB/RGBA 图像自动转换为灰度（取前三通道均值）
- 退化公式：`I_deg = T_total × (I ⊗ PSF) + L_path`
- 若不指定 `--output`，输出文件名自动由输入文件名生成，例如 `image.npy` → `image_degraded.npy`

---

## 物理模型说明

### 大气 PSF 定义与物理意义

大气点扩散函数（Atmospheric Point Spread Function）描述点光源经大气传播后在成像平面上的光强空间分布：

$$\text{PSF}(x, y) = \frac{\text{权重加权光子数}(x,y)}{N_\text{photons} \cdot \Delta A}$$

PSF 满足归一化条件：$\iint \text{PSF}(x,y)\,dx\,dy = 1$

### 蒙特卡洛光子追踪原理

**自由程采样**（Beer-Lambert 定律）：

$$s = -\frac{\ln R}{\sigma_t}, \quad R \sim U(0,1)$$

**权重修正法**（代替随机吸收）：

$$w \leftarrow w \cdot \omega_0, \quad \omega_0 = \frac{\sigma_\text{sca}}{\sigma_t}$$

**Henyey-Greenstein 相函数散射角采样**：

$$\cos\theta = \frac{1}{2g}\left[1 + g^2 - \left(\frac{1-g^2}{1-g+2gR}\right)^2\right]$$

**俄罗斯轮盘赌截断**（权重阈值 $10^{-3}$，存活概率 0.1）

### 退化模型

$$I_\text{deg}(x,y) = T_\text{total} \cdot \left[I_\text{scene}(x,y) \otimes \text{PSF}(x,y)\right] + L_\text{path}$$

其中路径辐射 $L_\text{path} = \varepsilon \cdot B(\lambda, T_\text{atm})$，$B$ 为普朗克黑体辐射函数。

### 透过率分解

$$T_\text{total} = T_\text{ballistic} + T_\text{scatter}$$

- **弹道透过率**：$T_\text{ballistic} \approx e^{-\sigma_t L}$（未经散射光子的贡献）
- **散射透过率**：$T_\text{scatter}$（经过至少一次散射的光子的贡献）

### MODTRAN5 参数提取与 g 优化流程

1. 解析 `.tp7` 文件，在目标波数处提取 `TOT_TRANS` → 计算 $\sigma_t$
2. 提取 `SING_SCAT` → 估算 $\omega_0$
3. 以最小化 $(T_\text{scatter,MC} - T_\text{scatter,MODTRAN})^2$ 为目标，用 `minimize_scalar` 优化 $g$

---

## MODTRAN5 文件格式说明

### .tp7 文件

固定列宽文本格式，`!` 开头为注释行，包含列头行（含 `FREQ` 关键字）：

```
FREQ    TOT_TRANS  PTH_THRML  THRML_SCT  SURF_EMIS  SOL_SCAT  SING_SCAT  ...
 1000.0  0.600000   0.025000   0.012000   ...
```

### .tp6（TAPE6）文件

详细逐层大气参数报告，包含高度、消光/散射/吸收系数等。

### .plt 文件

两列数值格式（波数 + 辐亮度），`!` 开头为注释行：

```
! wavenumber  radiance
  1000.0  0.035
```

---

## 模块 API 文档

### `AtmosphereParams`

大气参数数据类，包含 `sigma_t`、`omega_0`、`g`、`L`、`wavelength_um`、`T_atm`、`emissivity`。

预设工厂函数：`thin_fog(wavelength_um, L)`、`medium_fog(wavelength_um, L)`、`thick_fog(wavelength_um, L)`

### `MCSimulator`

```python
sim = MCSimulator(N_photons=2_000_000, bins=256, fov_radius=0.05, seed=42)
result = sim.simulate(params)  # 返回 psf, T_total, T_ballistic, T_scatter, ...
```

### `PSFAnalyzer`

```python
PSFAnalyzer.normalize(psf_raw, pixel_area)          # 归一化
PSFAnalyzer.resize_to_image(psf, target_shape)       # 缩放到图像分辨率
PSFAnalyzer.compute_encircled_energy(psf, xe, ye, radii)  # 包围能量
PSFAnalyzer.compute_fwhm(psf, xe, ye)               # 半高全宽
PSFAnalyzer.plot_psf(psf, xe, ye, save_path=...)    # 可视化
```

### `ImageDegradation`

```python
degrader = ImageDegradation()
degraded = degrader.degrade(image, psf_norm, T_total, L_path)
L = ImageDegradation.blackbody_radiance(T_K, wavelength_um)
L_path = ImageDegradation.compute_path_radiance(wavelength_um, T_atm, emissivity)
```

### `MODTRAN5Parser`

```python
parser = MODTRAN5Parser()
tp7_df = parser.parse_tp7('output.tp7')
plt_df = parser.parse_plt('output.plt')
params = parser.extract_mc_params(tp7_df, wavelength_um=10.0, path_length_km=1.0)
L_path = parser.get_path_radiance(plt_df, wavelength_um=10.0)
```

### `MODTRANOptimizer`

```python
opt = MODTRANOptimizer()
g_opt = opt.optimize_g(sigma_t, omega_0, L, T_scatter_target, mc_simulator)
best_params = opt.optimize_all(modtran_params, L, wavelength_um, mc_simulator)
validation = opt.validate(mc_result, modtran_params)
```

---

## 参数参考表

| 云雾类型 | sigma_t (km⁻¹) | omega_0 | g 典型范围 | 能见度 |
|---------|---------------|---------|-----------|--------|
| 薄雾（thin） | 0.5 | 0.85 | 0.70~0.80 | 2~5 km |
| 中雾（medium） | 2.0 | 0.90 | 0.75~0.85 | 0.5~2 km |
| 浓雾/云（thick） | 8.0 | 0.92 | 0.80~0.90 | < 0.5 km |

---

## 项目结构

```
MK/
├── README.md
├── requirements.txt
├── main.py
├── atmospheric_mc/
│   ├── __init__.py
│   ├── atmosphere.py          大气参数数据类与云雾预设
│   ├── mc_simulator.py        核心蒙特卡洛光子追踪
│   ├── psf.py                 PSF 分析工具
│   ├── image_degradation.py   图像退化模型
│   ├── modtran5_parser.py     MODTRAN5 文件解析器
│   └── optimizer.py           参数优化器
└── tests/
    ├── __init__.py
    ├── test_mc_simulator.py
    ├── test_psf.py
    ├── test_modtran5_parser.py
    └── test_image_degradation.py
```
