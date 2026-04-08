"""
红外图像大气退化模型。

实现基于卷积的图像退化，结合大气 PSF、透过率和路径辐射，
模拟云雾天气对红外成像系统的影响。
"""

import os
import numpy as np
from scipy.signal import fftconvolve
from .psf import PSFAnalyzer

# 支持的图像扩展名
_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
_SUPPORTED_IMAGE_EXTS = {'.npy'} | _IMG_EXTS
# 数值稳定性小量，防止除以零
_EPSILON = 1e-30


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
        psf_resized = psf_resized / (psf_sum + _EPSILON)

        # FFT 卷积（mode='same' 保持输出形状与输入一致）
        convolved = fftconvolve(image, psf_resized, mode='same')

        # 退化模型：I_deg = T_total * (I ⊗ PSF) + L_path
        degraded = T_total * convolved + L_path

        return degraded

    # ──────────────────────────────────────────────────────────
    # 文件级辅助接口
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        从文件加载图像并转换为 float64 灰度数组。

        支持格式：.npy、.png、.jpg、.jpeg、.tif、.tiff

        若为 RGB/RGBA 图像，自动取前三通道均值转换为灰度。

        Args:
            path: 图像文件路径

        Returns:
            float64 类型的二维灰度图像数组

        Raises:
            ValueError: 若文件扩展名不受支持
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            img = np.load(path)
        elif ext in _IMG_EXTS:
            from PIL import Image
            pil_img = Image.open(path)
            img = np.asarray(pil_img)
        else:
            raise ValueError(
                f'不支持的图像格式：{ext}，'
                f'支持：{", ".join(sorted(_SUPPORTED_IMAGE_EXTS))}'
            )
        img = np.asarray(img, dtype=np.float64)
        if img.ndim == 3:
            # 取前三通道均值转灰度（兼容 RGBA）
            img = img[..., :3].mean(axis=2)
        return img

    @staticmethod
    def load_psf(path: str) -> np.ndarray:
        """
        从文件加载 PSF，并做非负截断与归一化。

        支持格式：.npy、.png、.jpg、.jpeg、.tif、.tiff

        Args:
            path: PSF 文件路径

        Returns:
            归一化且非负的 float64 二维 PSF 数组（积分 = 1）

        Raises:
            ValueError: 若文件扩展名不受支持
        """
        psf = ImageDegradation.load_image(path)
        psf = np.maximum(psf, 0.0)
        psf_sum = psf.sum()
        psf = psf / (psf_sum + _EPSILON)
        return psf

    @staticmethod
    def save_image(path: str, image: np.ndarray) -> None:
        """
        将图像数组保存到文件。

        - 若扩展名为 .npy，直接保存原始数组；
        - 若为图像格式（.png / .jpg / .tif 等），先将值裁剪到
          [0, 1] 范围（按 min-max 归一化），再转为 uint8 保存。

        Args:
            path: 输出文件路径
            image: 待保存的二维图像数组

        Raises:
            ValueError: 若文件扩展名不受支持
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            np.save(path, image)
        elif ext in _IMG_EXTS:
            from PIL import Image
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                normalized = (image - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(image)
            uint8_img = (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)
            pil_img = Image.fromarray(uint8_img, mode='L')
            pil_img.save(path)
        else:
            raise ValueError(
                f'不支持的输出格式：{ext}，'
                f'支持：{", ".join(sorted(_SUPPORTED_IMAGE_EXTS))}'
            )

    def degrade_from_files(
        self,
        image_path: str,
        psf_path: str,
        t_total: float = 1.0,
        l_path: float = 0.0,
        output_path: str = None,
    ):
        """
        从文件直接完成图像卷积退化，并将结果保存到文件。

        完整流程：
        1. 加载图像（load_image）
        2. 加载并归一化 PSF（load_psf）
        3. 调用 degrade 完成退化
        4. 保存结果（save_image）

        若未提供 output_path，则根据 image_path 自动生成输出文件名：
        例如 ``image.npy`` → ``image_degraded.npy``

        Args:
            image_path: 输入图像文件路径
            psf_path: 输入 PSF 文件路径
            t_total: 大气总透过率，默认 1.0
            l_path: 路径辐射常量项，默认 0.0
            output_path: 输出文件路径；若为 None 则自动生成

        Returns:
            tuple(np.ndarray, str)：退化后的图像数组和最终输出路径
        """
        image = self.load_image(image_path)
        psf = self.load_psf(psf_path)

        degraded = self.degrade(image, psf, T_total=t_total, L_path=l_path)

        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = base + '_degraded' + ext

        self.save_image(output_path, degraded)
        return degraded, output_path

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
        if T_K <= 0 or wavelength_um <= 0:
            return 0.0
        lam = wavelength_um
        exponent = c2 / (lam * T_K)
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
