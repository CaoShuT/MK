"""
PSF 分析工具类。

提供大气点扩散函数的归一化、缩放、能量分析、FWHM 计算和可视化功能。
"""

import numpy as np
from scipy.ndimage import zoom


class PSFAnalyzer:
    """
    大气 PSF 分析与处理工具。

    提供 PSF 归一化、尺寸调整、包围能量、半高全宽（FWHM）
    计算及可视化等功能。
    """

    @staticmethod
    def normalize(psf_raw: np.ndarray, pixel_area: float) -> np.ndarray:
        """
        归一化 PSF 使其面积积分等于 1。

        归一化公式：PSF_norm = PSF_raw / (sum(PSF_raw) * pixel_area)

        Args:
            psf_raw: 原始 PSF 直方图，shape (H, W)
            pixel_area: 每个像素对应的物理面积 [km²]

        Returns:
            归一化后的 PSF，积分（sum * pixel_area）= 1
        """
        total = psf_raw.sum() * pixel_area
        return psf_raw / (total + 1e-30)

    @staticmethod
    def resize_to_image(psf: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        将 PSF 缩放到目标图像分辨率，缩放后重新归一化。

        使用 scipy.ndimage.zoom 进行双线性插值缩放。

        Args:
            psf: 输入 PSF，shape (H, W)
            target_shape: 目标形状 (H_target, W_target)

        Returns:
            缩放并重新归一化的 PSF，shape target_shape，积分=1
        """
        zoom_factors = (
            target_shape[0] / psf.shape[0],
            target_shape[1] / psf.shape[1],
        )
        psf_resized = zoom(psf, zoom_factors, order=1)
        # 确保非负
        psf_resized = np.maximum(psf_resized, 0.0)
        # 重新归一化（像素积分为1）
        total = psf_resized.sum()
        return psf_resized / (total + 1e-30)

    @staticmethod
    def compute_encircled_energy(
        psf: np.ndarray,
        xe: np.ndarray,
        ye: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        """
        计算不同半径内的包围能量占比（Encircled Energy）。

        Args:
            psf: 归一化 PSF，shape (H, W)
            xe: x 轴像素边界数组，长度 H+1
            ye: y 轴像素边界数组，长度 W+1
            radii: 待计算的半径数组 [km]

        Returns:
            各半径对应的能量占比数组，shape = radii.shape
        """
        # 像素中心坐标
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        XX, YY = np.meshgrid(xc, yc, indexing='ij')
        r_grid = np.sqrt(XX ** 2 + YY ** 2)

        total = psf.sum()
        ee = np.array([
            psf[r_grid <= r].sum() / (total + 1e-30)
            for r in radii
        ])
        return ee

    @staticmethod
    def compute_fwhm(
        psf: np.ndarray,
        xe: np.ndarray,
        ye: np.ndarray,
    ) -> float:
        """
        沿中心行计算 PSF 的半高全宽（FWHM）。

        方法：取 PSF 中心行的一维截面，找到峰值一半处用线性插值
        确定左右边界，FWHM = 右边界 - 左边界。

        Args:
            psf: 归一化 PSF，shape (H, W)
            xe: x 轴像素边界数组，长度 H+1
            ye: y 轴像素边界数组，长度 W+1

        Returns:
            FWHM 值 [km]，若无法计算则返回 -1.0
        """
        xc = 0.5 * (xe[:-1] + xe[1:])
        mid_col = psf.shape[1] // 2
        profile = psf[:, mid_col]
        half_max = profile.max() / 2.0

        if half_max <= 0:
            return -1.0

        # 找左侧越过半高的位置（从左向右扫描，找第一个上升过半高的点）
        left = None
        for i in range(len(profile) - 1):
            if profile[i] <= half_max <= profile[i + 1]:
                # 线性插值
                delta = profile[i + 1] - profile[i]
                if abs(delta) < 1e-30:
                    # 平坦段，取中点
                    left = 0.5 * (xc[i] + xc[i + 1])
                else:
                    t = (half_max - profile[i]) / delta
                    left = xc[i] + t * (xc[i + 1] - xc[i])
                break

        # 找右侧越过半高的位置（从右向左扫描，找第一个下降过半高的点）
        right = None
        for i in range(len(profile) - 1, 0, -1):
            if profile[i - 1] >= half_max >= profile[i]:
                delta = profile[i] - profile[i - 1]
                if abs(delta) < 1e-30:
                    right = 0.5 * (xc[i - 1] + xc[i])
                else:
                    t = (half_max - profile[i - 1]) / delta
                    right = xc[i - 1] + t * (xc[i] - xc[i - 1])
                break

        if left is None or right is None:
            return -1.0

        return float(right - left)

    @staticmethod
    def plot_psf(
        psf: np.ndarray,
        xe: np.ndarray,
        ye: np.ndarray,
        title: str = 'Atmospheric PSF',
        save_path: str = None,
    ):
        """
        可视化大气 PSF（线性图和对数图）。

        左图：线性尺度 PSF
        右图：log1p 尺度 PSF（增强弱散射翼的可见性）

        Args:
            psf: 归一化 PSF，shape (H, W)
            xe: x 轴像素边界数组
            ye: y 轴像素边界数组
            title: 图像标题
            save_path: 若提供则保存为文件，否则调用 plt.show()
        """
        import matplotlib.pyplot as plt

        extent = [ye[0], ye[-1], xe[0], xe[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].imshow(psf, extent=extent, origin='lower', cmap='hot', aspect='auto')
        axes[0].set_title(f'{title} (Linear)')
        axes[0].set_xlabel('y [km]')
        axes[0].set_ylabel('x [km]')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            np.log1p(psf), extent=extent, origin='lower', cmap='hot', aspect='auto'
        )
        axes[1].set_title(f'{title} (log1p)')
        axes[1].set_xlabel('y [km]')
        axes[1].set_ylabel('x [km]')
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def decompose(
        x_hits: np.ndarray,
        y_hits: np.ndarray,
        w_hits: np.ndarray,
        sca_counts: np.ndarray,
        bins: int,
        fov_radius: float,
    ) -> tuple:
        """
        将 PSF 分解为弹道分量和散射分量。

        弹道光子（scatter_count == 0）对应未散射光子，
        散射光子（scatter_count > 0）对应经历至少一次散射的光子。

        Args:
            x_hits: 击中点 x 坐标数组
            y_hits: 击中点 y 坐标数组
            w_hits: 对应权重数组
            sca_counts: 各光子的散射次数数组
            bins: 直方图分辨率
            fov_radius: 视场半径 [km]

        Returns:
            (psf_ballistic, psf_scatter)：各分量的归一化 PSF 数组
        """
        R = fov_radius
        hist_range = [[-R, R], [-R, R]]
        pixel_area = (2.0 * R / bins) ** 2

        is_ball = sca_counts == 0

        ball_hist, xe, ye = np.histogram2d(
            x_hits[is_ball] if np.any(is_ball) else np.array([]),
            y_hits[is_ball] if np.any(is_ball) else np.array([]),
            bins=bins,
            weights=w_hits[is_ball] if np.any(is_ball) else np.array([]),
            range=hist_range,
        )
        sca_hist, _, _ = np.histogram2d(
            x_hits[~is_ball] if np.any(~is_ball) else np.array([]),
            y_hits[~is_ball] if np.any(~is_ball) else np.array([]),
            bins=bins,
            weights=w_hits[~is_ball] if np.any(~is_ball) else np.array([]),
            range=hist_range,
        )

        total = (ball_hist.sum() + sca_hist.sum()) * pixel_area
        psf_ballistic = ball_hist / (total + 1e-30)
        psf_scatter = sca_hist / (total + 1e-30)

        return psf_ballistic, psf_scatter
