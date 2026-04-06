"""
ImageDegradation 单元测试。

验证图像退化卷积、能量守恒、黑体辐射计算等功能。
"""

import numpy as np
import pytest

from atmospheric_mc import ImageDegradation, PSFAnalyzer


def make_delta_psf(size=32):
    """创建归一化 delta 函数 PSF（理想无失真情形）。"""
    psf = np.zeros((size, size))
    psf[size // 2, size // 2] = 1.0
    return psf


def make_gaussian_psf_normalized(size=32, sigma_px=3.0):
    """创建归一化高斯 PSF。"""
    cx, cy = size // 2, size // 2
    x = np.arange(size)
    y = np.arange(size)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    psf = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma_px ** 2))
    return psf / psf.sum()


class TestDegradeOutputShape:
    """测试卷积输出形状与输入一致。"""

    def test_square_image(self):
        """方形图像输出形状验证。"""
        image = np.random.RandomState(0).rand(64, 64)
        psf = make_gaussian_psf_normalized(32, sigma_px=3.0)
        degrader = ImageDegradation()

        result = degrader.degrade(image, psf, T_total=1.0, L_path=0.0)
        assert result.shape == image.shape, (
            f'输出形状错误：{result.shape}，期望 {image.shape}'
        )

    def test_rectangular_image(self):
        """矩形图像输出形状验证。"""
        image = np.random.RandomState(1).rand(64, 128)
        psf = make_gaussian_psf_normalized(32, sigma_px=3.0)
        degrader = ImageDegradation()

        result = degrader.degrade(image, psf, T_total=0.8, L_path=0.1)
        assert result.shape == image.shape, (
            f'矩形图像输出形状错误：{result.shape}，期望 {image.shape}'
        )

    def test_large_image(self):
        """大尺寸图像（512x512）形状验证。"""
        image = np.ones((512, 512))
        psf = make_gaussian_psf_normalized(64, sigma_px=5.0)
        degrader = ImageDegradation()

        result = degrader.degrade(image, psf, T_total=0.5)
        assert result.shape == (512, 512)


class TestEnergyConservation:
    """测试 T_total=1, L_path=0 时能量基本守恒。

    使用 delta PSF（理想情形），输出应与输入近似相等（误差 < 5%）。
    """

    def test_energy_conservation_delta_psf(self):
        """Delta PSF + T=1 + L=0：输出与输入能量之比接近 1。"""
        np.random.seed(42)
        image = np.abs(np.random.randn(64, 64)) + 1.0
        psf = make_delta_psf(32)
        degrader = ImageDegradation()

        result = degrader.degrade(image, psf, T_total=1.0, L_path=0.0)

        # 能量比
        energy_ratio = result.sum() / (image.sum() + 1e-30)
        assert abs(energy_ratio - 1.0) < 0.05, (
            f'能量守恒误差超标：比值={energy_ratio:.4f}，期望接近 1.0'
        )

    def test_transmittance_scaling(self):
        """T_total 缩放验证：退化图像均值 ≈ T_total * 原始均值（delta PSF）。"""
        image = np.ones((32, 32)) * 100.0
        psf = make_delta_psf(16)
        degrader = ImageDegradation()
        T = 0.6

        result = degrader.degrade(image, psf, T_total=T, L_path=0.0)

        # 排除边缘效应，检查中心区域
        center = result[8:24, 8:24]
        expected = T * 100.0
        rel_err = abs(center.mean() - expected) / expected
        assert rel_err < 0.05, (
            f'T_total 缩放误差：均值={center.mean():.2f}，期望={expected:.2f}'
        )

    def test_path_radiance_offset(self):
        """L_path 偏移验证：退化结果应包含路径辐射偏置。"""
        image = np.zeros((32, 32))
        psf = make_delta_psf(16)
        degrader = ImageDegradation()
        L_path = 5.0

        result = degrader.degrade(image, psf, T_total=1.0, L_path=L_path)

        # 全零图像退化后应约等于 L_path
        assert abs(result.mean() - L_path) < 0.5, (
            f'路径辐射偏移错误：均值={result.mean():.2f}，期望={L_path:.2f}'
        )


class TestBlackbodyRadiance:
    """测试黑体辐射计算。"""

    def test_positive_value(self):
        """T=300K, lambda=10μm 应返回正值。"""
        val = ImageDegradation.blackbody_radiance(T_K=300.0, wavelength_um=10.0)
        assert val > 0, f'黑体辐亮度应为正值，实际={val}'

    def test_higher_temp_higher_radiance(self):
        """温度更高时辐亮度应更大（同一波长）。"""
        val1 = ImageDegradation.blackbody_radiance(T_K=300.0, wavelength_um=10.0)
        val2 = ImageDegradation.blackbody_radiance(T_K=400.0, wavelength_um=10.0)
        assert val2 > val1, (
            f'400K 辐亮度应大于 300K：val2={val2:.2f}, val1={val1:.2f}'
        )

    def test_known_value_8um(self):
        """8 μm 处 300K 黑体辐亮度应为正且数量级合理。"""
        val = ImageDegradation.blackbody_radiance(T_K=300.0, wavelength_um=8.0)
        assert val > 0
        # 使用 c1=1.19104e8, c2=1.43878e4 的单位体系（W/(m²·sr·μm)），
        # 300K @ 8μm 典型值约 9 W/(m²·sr·μm)
        assert 1.0 < val < 1e4, f'数量级异常：{val:.2e}'

    def test_compute_path_radiance(self):
        """路径辐射 = emissivity * 黑体辐亮度。"""
        T_atm = 280.0
        wavelength_um = 10.0
        emissivity = 0.6

        L_path = ImageDegradation.compute_path_radiance(wavelength_um, T_atm, emissivity)
        B = ImageDegradation.blackbody_radiance(T_atm, wavelength_um)
        expected = emissivity * B

        assert abs(L_path - expected) < 1e-10, (
            f'路径辐射计算错误：{L_path:.4f}，期望 {expected:.4f}'
        )
