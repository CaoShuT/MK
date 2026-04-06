"""
PSFAnalyzer 单元测试。

验证 PSF 归一化、尺寸调整和 FWHM 计算功能的正确性。
"""

import numpy as np
import pytest

from atmospheric_mc import PSFAnalyzer


def make_gaussian_psf(size=64, sigma_px=5.0):
    """生成归一化高斯 PSF 用于测试。"""
    cx, cy = size // 2, size // 2
    x = np.arange(size)
    y = np.arange(size)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    psf = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma_px ** 2))
    return psf


class TestNormalize:
    """测试1：normalize 后积分接近 1。"""

    def test_normalize_uniform(self):
        """均匀分布 PSF 归一化测试。"""
        psf_raw = np.ones((32, 32))
        pixel_area = 1.0
        psf_norm = PSFAnalyzer.normalize(psf_raw, pixel_area)
        integral = psf_norm.sum() * pixel_area
        assert abs(integral - 1.0) < 1e-10, f'归一化积分误差：{integral}'

    def test_normalize_gaussian(self):
        """高斯 PSF 归一化测试，像素面积 = (0.1km)²。"""
        psf_raw = make_gaussian_psf(64, sigma_px=5.0)
        pixel_area = (0.001) ** 2  # 0.001 km 像素
        psf_norm = PSFAnalyzer.normalize(psf_raw, pixel_area)
        integral = psf_norm.sum() * pixel_area
        assert abs(integral - 1.0) < 1e-10, f'高斯 PSF 归一化积分误差：{integral}'

    def test_normalize_output_nonnegative(self):
        """归一化输出应全为非负值。"""
        psf_raw = np.abs(np.random.RandomState(0).randn(32, 32))
        psf_norm = PSFAnalyzer.normalize(psf_raw, 1.0)
        assert np.all(psf_norm >= 0)


class TestResizeToImage:
    """测试2：resize_to_image 后形状正确且积分归一化。"""

    def test_resize_shape(self):
        """验证缩放输出形状正确。"""
        psf = make_gaussian_psf(32, sigma_px=3.0)
        psf = psf / psf.sum()  # 像素求和=1

        target = (128, 128)
        psf_resized = PSFAnalyzer.resize_to_image(psf, target)
        assert psf_resized.shape == target, (
            f'缩放形状错误：{psf_resized.shape}，期望 {target}'
        )

    def test_resize_normalization(self):
        """缩放后 PSF 像素求和应为 1（归一化）。"""
        psf = make_gaussian_psf(64, sigma_px=5.0)
        psf = psf / psf.sum()

        for target in [(32, 32), (128, 128), (256, 512)]:
            psf_r = PSFAnalyzer.resize_to_image(psf, target)
            s = psf_r.sum()
            assert abs(s - 1.0) < 1e-6, (
                f'目标形状 {target} 缩放后归一化误差：{abs(s - 1.0):.2e}'
            )

    def test_resize_nonnegative(self):
        """缩放后 PSF 应全为非负值。"""
        psf = make_gaussian_psf(32, sigma_px=3.0)
        psf = psf / psf.sum()
        psf_r = PSFAnalyzer.resize_to_image(psf, (64, 64))
        assert np.all(psf_r >= 0)


class TestComputeFWHM:
    """测试3：compute_fwhm 对已知高斯 PSF 返回正确值。"""

    def test_fwhm_gaussian(self):
        """已知高斯 sigma 的 FWHM 验证。

        理论关系：FWHM = 2√(2ln2) * sigma ≈ 2.3548 * sigma
        """
        size = 256
        sigma_px = 10.0
        psf = make_gaussian_psf(size, sigma_px=sigma_px)
        psf = psf / psf.sum()

        # 构建像素坐标边界（每像素 0.001 km）
        dx = 0.001  # km
        fov = size * dx
        xe = np.linspace(-fov / 2, fov / 2, size + 1)
        ye = np.linspace(-fov / 2, fov / 2, size + 1)

        fwhm = PSFAnalyzer.compute_fwhm(psf, xe, ye)

        # 理论 FWHM（km）
        sigma_km = sigma_px * dx
        fwhm_expected = 2.3548 * sigma_km

        rel_err = abs(fwhm - fwhm_expected) / fwhm_expected
        assert rel_err < 0.05, (
            f'FWHM 误差超标：计算值={fwhm:.5f} km, '
            f'理论值={fwhm_expected:.5f} km, 相对误差={rel_err:.2%}'
        )

    def test_fwhm_sharp_peak(self):
        """单像素峰 PSF 的 FWHM 应极小。"""
        size = 64
        psf = np.zeros((size, size))
        psf[size // 2, size // 2] = 1.0

        dx = 0.001
        fov = size * dx
        xe = np.linspace(-fov / 2, fov / 2, size + 1)
        ye = np.linspace(-fov / 2, fov / 2, size + 1)

        fwhm = PSFAnalyzer.compute_fwhm(psf, xe, ye)
        # 单像素峰返回 -1 或极小值均可接受
        assert fwhm < dx * 2 or fwhm == -1.0, f'单像素 FWHM 异常：{fwhm}'
