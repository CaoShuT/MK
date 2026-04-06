"""
红外图像大气退化模型。

实现基于卷积的图像退化，结合大气 PSF、透过率和路径辐射，
模拟云雾天气对红外成像系统的影响。
"""

import numpy as np
from scipy.signal import fftconvolve
from .psf import PSFAnalyzer


class ImageDegradation:
    """
    红外图像大气传输退化处理器。

    退化模型：
        I_deg(x,y) = T_total * [I_scene(x,y) ⊗ PSF_norm(x,y)] + L_path

    其中：
        - T_total：大气总透过率
        - PSF_norm：归一化大气点扩散函数（积分=1）
        - L_path：大气路径辐射（背景辐射）
        - ⊗：二维卷积运算
    """

    def degrade(
        self,
        image: np.ndarray,
        psf_norm: np.ndarray,
        T_total: float,
        L_path: float = 0.0,
    ) -> np.ndarray:
        """
        对输入图像施加大气退化效应。

        处理步骤：
        1. 将 PSF 缩放到图像分辨率
        2. 确保 PSF 非负并重新归一化
        3. 用 FFT 卷积模拟大气模糊
        4. 乘以透过率并叠加路径辐射

        Args:
            image: 输入红外场景图像，shape (H, W)，单位为辐亮度
            psf_norm: 归一化大气 PSF，shape 任意（将被缩放到 image.shape）
            T_total: 大气总透过率，取值 [0, 1]
            L_path: 大气路径辐射 [W/(m²·sr·μm)]，默认 0.0

        Returns:
            退化后的红外图像，shape 与 image 相同
        """
        # 将 PSF 缩放到图像分辨率
        psf_resized = PSFAnalyzer.resize_to_image(psf_norm, image.shape)

        # 确保非负并归一化
        psf_resized = np.maximum(psf_resized, 0.0)
        psf_sum = psf_resized.sum()
        psf_resized = psf_resized / (psf_sum + 1e-30)

        # FFT 卷积（mode='same' 保持输出形状与输入一致）
        convolved = fftconvolve(image, psf_resized, mode='same')

        # 退化模型：I_deg = T_total * (I ⊗ PSF) + L_path
        degraded = T_total * convolved + L_path

        return degraded

    @staticmethod
    def blackbody_radiance(T_K: float, wavelength_um: float) -> float:
        """
        计算普朗克黑体辐射辐亮度。

        普朗克公式（第一辐射常数和第二辐射常数使用红外常用单位）：
            B(λ, T) = c1 / (λ⁵ * (exp(c2 / (λ*T)) - 1))

        其中：
            c1 = 1.19104e8  [W·μm⁴/(m²·sr)]
            c2 = 1.43878e4  [μm·K]

        Args:
            T_K: 温度 [K]
            wavelength_um: 波长 [μm]

        Returns:
            黑体辐亮度 [W/(m²·sr·μm)]
        """
        c1 = 1.19104e8   # W·μm⁴/(m²·sr)
        c2 = 1.43878e4   # μm·K
        lam = wavelength_um
        exponent = c2 / (lam * T_K + 1e-30)
        # 防止溢出
        exponent = np.clip(exponent, 0.0, 700.0)
        return c1 / (lam ** 5 * (np.exp(exponent) - 1.0 + 1e-30))

    @staticmethod
    def compute_path_radiance(
        wavelength_um: float,
        T_atm: float,
        emissivity: float,
    ) -> float:
        """
        计算大气路径辐射（自发辐射贡献）。

        路径辐射 = emissivity * B(T_atm, lambda)

        Args:
            wavelength_um: 工作波长 [μm]
            T_atm: 大气等效温度 [K]
            emissivity: 大气有效发射率，取值 [0, 1]

        Returns:
            路径辐射值 [W/(m²·sr·μm)]
        """
        return emissivity * ImageDegradation.blackbody_radiance(T_atm, wavelength_um)
