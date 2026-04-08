"""
红外图像大气传输效应蒙特卡洛仿真系统 - 主入口脚本。

支持三种运行模式：
  mc       纯蒙特卡洛演示（无需 MODTRAN5 文件）
  modtran  基于 MODTRAN5 输出的优化仿真
  convolve 将已有图像与已有 PSF 直接做卷积退化

用法示例：
    python main.py --mode mc --fog medium --wavelength 10.0 --distance 1.0 --n_photons 1000000
    python main.py --mode modtran --tp7 file.tp7 --plt file.plt --wavelength 10.0 --distance 1.0
    python main.py --mode modtran --tp7 file.tp7 --plt file.plt --image scene.npy
    python main.py --mode convolve --image image.npy --psf psf.npy
    python main.py --mode convolve --image image.png --psf psf.npy --t_total 0.75 --l_path 0.02 --output result.npy
    python main.py --mode convolve --image image.png --psf psf.npy --output result.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from atmospheric_mc import (
    AtmosphereParams,
    MCSimulator,
    PSFAnalyzer,
    ImageDegradation,
    MODTRAN5Parser,
    MODTRANOptimizer,
    thin_fog,
    medium_fog,
    thick_fog,
)


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────

def make_synthetic_scene(size: int = 512, wavelength_um: float = 10.0) -> np.ndarray:
    """
    生成合成红外场景图像。

    包含多个高斯热目标叠加在背景辐射之上。

    Args:
        size: 图像尺寸（正方形），默认 512
        wavelength_um: 工作波长 [μm]，用于计算黑体辐亮度背景

    Returns:
        合成场景图像 np.ndarray，shape (size, size)
    """
    # 背景：300K 黑体辐亮度
    T_bg = 300.0
    L_bg = ImageDegradation.blackbody_radiance(T_bg, wavelength_um)

    scene = np.full((size, size), L_bg, dtype=np.float64)

    # 添加热目标（高斯分布）
    targets = [
        (size // 4,     size // 4,     20.0, 15.0),   # (行, 列, sigma, delta_T)
        (size // 2,     size // 2,     30.0, 25.0),
        (3 * size // 4, size // 3,     15.0, 20.0),
        (size // 3,     3 * size // 4, 10.0, 30.0),
    ]

    xx = np.arange(size)
    yy = np.arange(size)
    XX, YY = np.meshgrid(xx, yy, indexing='ij')

    for (cx, cy, sigma, delta_T) in targets:
        T_hot = T_bg + delta_T
        L_hot = ImageDegradation.blackbody_radiance(T_hot, wavelength_um)
        gauss = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma ** 2))
        scene += (L_hot - L_bg) * gauss

    return scene


def save_comparison(original: np.ndarray, degraded: np.ndarray, save_path: str):
    """
    保存原始图像与退化图像的对比图。

    Args:
        original: 原始场景图像
        degraded: 退化后图像
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = min(original.min(), degraded.min())
    vmax = max(original.max(), degraded.max())

    im0 = axes[0].imshow(original, cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('原始红外场景图像')
    axes[0].set_xlabel('列（像素）')
    axes[0].set_ylabel('行（像素）')
    plt.colorbar(im0, ax=axes[0], label='辐亮度 [W/(m²·sr·μm)]')

    im1 = axes[1].imshow(degraded, cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('大气退化后图像')
    axes[1].set_xlabel('列（像素）')
    axes[1].set_ylabel('行（像素）')
    plt.colorbar(im1, ax=axes[1], label='辐亮度 [W/(m²·sr·μm)]')

    plt.suptitle('红外图像大气传输效应仿真结果', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] 退化对比图 → {save_path}')


# ─────────────────────────────────────────────────────────
# 模式A：纯蒙特卡洛演示
# ─────────────────────────────────────────────────────────

def run_pure_mc_demo(args):
    """
    模式A：纯蒙特卡洛仿真演示（无需 MODTRAN5 文件）。

    步骤：
    1. 根据 args.fog 选择预设大气参数
    2. 生成合成红外场景图像
    3. 运行 MC 仿真得到 PSF
    4. 计算 FWHM 及各透过率
    5. 对场景施加大气退化
    6. 保存 PSF 可视化图和退化对比图

    Args:
        args: argparse 命令行参数
    """
    print('='*60)
    print('模式A：纯蒙特卡洛大气 PSF 仿真')
    print('='*60)

    # 1. 选择云雾预设参数
    fog_map = {
        'thin':   thin_fog,
        'medium': medium_fog,
        'thick':  thick_fog,
    }
    if args.fog not in fog_map:
        raise ValueError(f'未知云雾类型：{args.fog}，可选：thin/medium/thick')

    params = fog_map[args.fog](wavelength_um=args.wavelength, L=args.distance)
    print(f'\n大气参数：')
    print(f'  云雾类型:    {args.fog}')
    print(f'  sigma_t:    {params.sigma_t:.3f} km⁻¹')
    print(f'  omega_0:    {params.omega_0:.3f}')
    print(f'  g:          {params.g:.3f}')
    print(f'  L:          {params.L:.3f} km')
    print(f'  波长:       {params.wavelength_um:.1f} μm')

    # 2. 生成合成场景图像
    print(f'\n生成合成红外场景图像（{args.scene_size}×{args.scene_size}）...')
    scene = make_synthetic_scene(size=args.scene_size, wavelength_um=args.wavelength)
    print(f'  场景辐亮度范围：[{scene.min():.2f}, {scene.max():.2f}] W/(m²·sr·μm)')

    # 3. 运行 MC 仿真
    print(f'\n运行蒙特卡洛仿真（N={args.n_photons:,}）...')
    sim = MCSimulator(
        N_photons=args.n_photons,
        bins=args.bins,
        fov_radius=args.fov_radius,
        seed=args.seed,
    )
    result = sim.simulate(params)

    # 4. 计算统计量
    psf = result['psf']
    xe = result['xe']
    ye = result['ye']
    fwhm = PSFAnalyzer.compute_fwhm(psf, xe, ye)

    print(f'\n仿真结果：')
    print(f'  T_total:     {result["T_total"]:.6f}')
    print(f'  T_ballistic: {result["T_ballistic"]:.6f}  （≈ exp(-σ_t·L) = {np.exp(-params.sigma_t*params.L):.6f}）')
    print(f'  T_scatter:   {result["T_scatter"]:.6f}')
    print(f'  FWHM:        {fwhm*1000:.2f} m  ({fwhm:.5f} km)')
    print(f'  击中光子数:  {result["n_hits"]:,}')

    # 5. 保存 PSF 可视化
    psf_save = 'psf_visualization.png'
    PSFAnalyzer.plot_psf(
        psf, xe, ye,
        title=f'大气 PSF（{args.fog}雾, σ_t={params.sigma_t}, L={params.L}km）',
        save_path=psf_save,
    )
    print(f'\n[保存] PSF 可视化图 → {psf_save}')

    # 6. 图像退化
    print('\n施加大气退化...')
    degrader = ImageDegradation()
    L_path = ImageDegradation.compute_path_radiance(
        params.wavelength_um, params.T_atm, params.emissivity
    )
    print(f'  路径辐射 L_path = {L_path:.2f} W/(m²·sr·μm)')

    degraded = degrader.degrade(scene, psf, T_total=result['T_total'], L_path=L_path)
    print(f'  退化图像辐亮度范围：[{degraded.min():.2f}, {degraded.max():.2f}]')

    save_comparison(scene, degraded, 'degradation_comparison.png')
    print('\n仿真完成！')


# ─────────────────────────────────────────────────────────
# 模式B：MODTRAN5 优化仿真
# ─────────────────────────────────────────────────────────

def run_modtran_optimized(args):
    """
    模式B：基于 MODTRAN5 输出文件的优化蒙特卡洛仿真。

    步骤：
    1. 解析 .tp7 和 .plt 文件
    2. 提取大气参数
    3. 优化 HG 不对称因子 g
    4. 运行高精度 MC 仿真
    5. 验证结果并退化图像

    Args:
        args: argparse 命令行参数
    """
    print('='*60)
    print('模式B：MODTRAN5 优化蒙特卡洛仿真')
    print('='*60)

    # 1. 解析 MODTRAN5 文件
    parser = MODTRAN5Parser()

    print(f'\n解析 .tp7 文件：{args.tp7}')
    tp7_df = parser.parse_tp7(args.tp7)
    print(f'  解析到 {len(tp7_df)} 行辐射传输数据')

    plt_df = None
    if args.plt:
        print(f'解析 .plt 文件：{args.plt}')
        plt_df = parser.parse_plt(args.plt)
        print(f'  解析到 {len(plt_df)} 行路径辐射数据')

    # 2. 提取 MC 参数
    print(f'\n提取蒙特卡洛参数（λ={args.wavelength}μm, L={args.distance}km）...')
    modtran_params = parser.extract_mc_params(tp7_df, args.wavelength, args.distance)

    print(f'  提取参数：')
    print(f'    sigma_t:     {modtran_params["sigma_t"]:.4f} km⁻¹')
    print(f'    omega_0:     {modtran_params["omega_0"]:.4f}')
    print(f'    T_total:     {modtran_params["T_total"]:.4f}')
    print(f'    T_scatter:   {modtran_params["T_scatter"]:.4f}')
    print(f'    波数(使用):  {modtran_params["wavenumber_used"]:.1f} cm⁻¹')

    # 3. 优化 g
    print(f'\n优化 HG 不对称因子 g...')
    opt_sim = MCSimulator(N_photons=args.n_photons, bins=args.bins,
                          fov_radius=args.fov_radius, seed=args.seed)
    optimizer = MODTRANOptimizer()
    g_opt = optimizer.optimize_g(
        sigma_t=modtran_params['sigma_t'],
        omega_0=modtran_params['omega_0'],
        L=args.distance,
        T_scatter_target=modtran_params['T_scatter'],
        mc_simulator=opt_sim,
        n_photons_opt=200_000,
    )

    # 4. 高精度最终仿真
    print(f'\n运行最终高精度仿真（N={args.n_photons:,}，g={g_opt:.4f}）...')
    final_params = AtmosphereParams(
        sigma_t=modtran_params['sigma_t'],
        omega_0=modtran_params['omega_0'],
        g=g_opt,
        L=args.distance,
        wavelength_um=args.wavelength,
        T_atm=280.0,
        emissivity=0.6,
    )
    final_sim = MCSimulator(N_photons=args.n_photons, bins=args.bins,
                            fov_radius=args.fov_radius, seed=args.seed)
    final_result = final_sim.simulate(final_params)

    # 5. 验证
    optimizer.validate(final_result, modtran_params)

    # PSF 统计
    fwhm = PSFAnalyzer.compute_fwhm(final_result['psf'], final_result['xe'], final_result['ye'])
    print(f'\nPSF FWHM: {fwhm*1000:.2f} m')

    # 保存 PSF 可视化
    PSFAnalyzer.plot_psf(
        final_result['psf'], final_result['xe'], final_result['ye'],
        title=f'优化后大气 PSF（MODTRAN5 辅助）',
        save_path='psf_modtran_optimized.png',
    )

    # 6. 加载或生成图像
    if args.image:
        print(f'\n加载图像：{args.image}')
        scene = np.load(args.image)
    else:
        print('\n生成合成红外场景图像...')
        scene = make_synthetic_scene(size=args.scene_size, wavelength_um=args.wavelength)

    # 退化
    degrader = ImageDegradation()
    L_path = 0.0
    if plt_df is not None and len(plt_df) > 0:
        L_path = parser.get_path_radiance(plt_df, args.wavelength)

    degraded = degrader.degrade(
        scene, final_result['psf'],
        T_total=final_result['T_total'],
        L_path=L_path,
    )
    save_comparison(scene, degraded, 'degradation_modtran.png')
    print('\nMODTRAN5 优化仿真完成！')


# ─────────────────────────────────────────────────────────
# 模式C：已有图像 + 已有 PSF 直接卷积退化
# ─────────────────────────────────────────────────────────

def run_convolve_mode(args):
    """
    模式C：将已有图像与已有 PSF 直接做卷积退化。

    步骤：
    1. 加载图像（支持 .npy / .png / .jpg / .jpeg / .tif / .tiff）
    2. 加载 PSF 并做非负截断与归一化
    3. 调用大气退化模型：I_deg = T_total * (I ⊗ PSF) + L_path
    4. 保存结果

    Args:
        args: argparse 命令行参数
    """
    print('='*60)
    print('模式C：已有图像与 PSF 直接卷积退化')
    print('='*60)

    degrader = ImageDegradation()

    print(f'\n加载图像：{args.image}')
    image = degrader.load_image(args.image)
    print(f'  图像尺寸：{image.shape}，值域：[{image.min():.4g}, {image.max():.4g}]')

    print(f'加载 PSF：{args.psf}')
    psf = degrader.load_psf(args.psf)
    print(f'  PSF 尺寸：{psf.shape}，归一化和：{psf.sum():.6f}')

    print(f'\n退化参数：T_total={args.t_total}, L_path={args.l_path}')
    degraded = degrader.degrade(image, psf, T_total=args.t_total, L_path=args.l_path)
    print(f'  退化图像尺寸：{degraded.shape}，值域：[{degraded.min():.4g}, {degraded.max():.4g}]')

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        output_path = base + '_degraded' + ext

    degrader.save_image(output_path, degraded)
    print(f'\n[保存] 退化结果 → {output_path}')
    print('\n卷积退化完成！')


# ─────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description='红外图像大气传输效应蒙特卡洛仿真系统',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--mode', choices=['mc', 'modtran', 'convolve'], default='mc',
        help=(
            '运行模式：mc=纯蒙特卡洛演示，modtran=MODTRAN5优化仿真，'
            'convolve=已有图像与PSF直接卷积退化'
        ),
    )
    # MC 参数
    parser.add_argument('--fog', choices=['thin', 'medium', 'thick'], default='medium',
                        help='云雾类型（仅 mc 模式）')
    parser.add_argument('--wavelength', type=float, default=10.0,
                        help='工作波长 [μm]')
    parser.add_argument('--distance', type=float, default=1.0,
                        help='大气路径长度 [km]')
    parser.add_argument('--n_photons', type=int, default=1_000_000,
                        help='模拟光子数量')
    parser.add_argument('--bins', type=int, default=256,
                        help='PSF 直方图分辨率')
    parser.add_argument('--fov_radius', type=float, default=0.05,
                        help='视场半径 [km]')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子')
    parser.add_argument('--scene_size', type=int, default=512,
                        help='合成场景图像尺寸（像素）')
    # MODTRAN5 参数
    parser.add_argument('--tp7', type=str, default=None,
                        help='.tp7 文件路径（modtran 模式必填）')
    parser.add_argument('--plt', type=str, default=None,
                        help='.plt 文件路径（可选）')
    # 图像路径（mc/modtran/convolve 共用）
    parser.add_argument('--image', type=str, default=None,
                        help=(
                            '输入图像文件路径（convolve 模式必填，modtran 模式可选）；'
                            '支持格式：.npy、.png、.jpg、.jpeg、.tif、.tiff'
                        ))
    # convolve 模式专用参数
    parser.add_argument('--psf', type=str, default=None,
                        help='输入 PSF 文件路径（convolve 模式必填）；支持格式：.npy、.png 等')
    parser.add_argument('--t_total', type=float, default=1.0,
                        help='大气总透过率，取值 [0, 1]（convolve 模式）')
    parser.add_argument('--l_path', type=float, default=0.0,
                        help='路径辐射常量项（convolve 模式）')
    parser.add_argument('--output', type=str, default=None,
                        help=(
                            '输出文件路径（convolve 模式）；'
                            '若未提供则自动生成（如 image_degraded.npy）'
                        ))
    return parser


def main():
    """主入口函数。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'mc':
        run_pure_mc_demo(args)
    elif args.mode == 'modtran':
        if args.tp7 is None:
            parser.error('--mode modtran 需要提供 --tp7 参数')
        run_modtran_optimized(args)
    elif args.mode == 'convolve':
        if args.image is None:
            parser.error('--mode convolve 需要提供 --image 参数')
        if args.psf is None:
            parser.error('--mode convolve 需要提供 --psf 参数')
        run_convolve_mode(args)


if __name__ == '__main__':
    main()
