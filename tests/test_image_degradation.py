"""
ImageDegradation 单元测试。

验证图像退化卷积、能量守恒、黑体辐射计算等功能。
"""

import os
import tempfile
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


class TestLoadImage:
    """测试 load_image 方法。"""

    def test_load_npy_image(self):
        """load_image 能正确读取 .npy 格式的图像。"""
        image = np.random.RandomState(10).rand(32, 32)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, image)
            path = f.name
        try:
            loaded = ImageDegradation.load_image(path)
            assert loaded.dtype == np.float64, '返回类型应为 float64'
            assert loaded.shape == (32, 32), f'形状错误：{loaded.shape}'
            np.testing.assert_allclose(loaded, image, rtol=1e-10)
        finally:
            os.unlink(path)

    def test_load_npy_image_dtype_float64(self):
        """load_image 返回值类型始终为 float64。"""
        image = np.ones((16, 16), dtype=np.float32) * 2.5
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, image)
            path = f.name
        try:
            loaded = ImageDegradation.load_image(path)
            assert loaded.dtype == np.float64
        finally:
            os.unlink(path)

    def test_load_png_image(self):
        """load_image 能读取 .png 格式的图像（灰度）。"""
        from PIL import Image as PilImage
        arr = (np.random.RandomState(7).rand(24, 24) * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            PilImage.fromarray(arr, mode='L').save(f.name)
            path = f.name
        try:
            loaded = ImageDegradation.load_image(path)
            assert loaded.dtype == np.float64
            assert loaded.shape == (24, 24)
        finally:
            os.unlink(path)

    def test_load_png_rgb_to_gray(self):
        """load_image 对 RGB 图像自动转为灰度。"""
        from PIL import Image as PilImage
        arr = (np.random.RandomState(3).rand(16, 16, 3) * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            PilImage.fromarray(arr, mode='RGB').save(f.name)
            path = f.name
        try:
            loaded = ImageDegradation.load_image(path)
            assert loaded.ndim == 2, '转换后应为二维灰度图'
        finally:
            os.unlink(path)

    def test_load_unsupported_format_raises(self):
        """load_image 对不支持的扩展名应抛出 ValueError。"""
        with pytest.raises(ValueError, match='不支持的图像格式'):
            ImageDegradation.load_image('image.bmp')


class TestLoadPSF:
    """测试 load_psf 方法。"""

    def test_psf_nonnegative(self):
        """load_psf 返回结果非负。"""
        psf_raw = np.array([[-1.0, 2.0], [3.0, -0.5]])
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, psf_raw)
            path = f.name
        try:
            psf = ImageDegradation.load_psf(path)
            assert (psf >= 0).all(), 'PSF 不应含负值'
        finally:
            os.unlink(path)

    def test_psf_normalized_sum_one(self):
        """load_psf 返回 PSF 之和应为 1。"""
        psf_raw = np.random.RandomState(5).rand(16, 16) + 0.1
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, psf_raw)
            path = f.name
        try:
            psf = ImageDegradation.load_psf(path)
            assert abs(psf.sum() - 1.0) < 1e-10, f'PSF 归一化和应为 1，实际为 {psf.sum()}'
        finally:
            os.unlink(path)


class TestSaveImage:
    """测试 save_image 方法。"""

    def test_save_npy(self):
        """save_image 保存 .npy 后可原样读回。"""
        image = np.random.RandomState(99).rand(20, 20)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            ImageDegradation.save_image(path, image)
            loaded = np.load(path)
            np.testing.assert_allclose(loaded, image)
        finally:
            os.unlink(path)

    def test_save_png(self):
        """save_image 保存 .png 后文件存在且可读回。"""
        from PIL import Image as PilImage
        image = np.random.RandomState(88).rand(24, 24) * 100
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            ImageDegradation.save_image(path, image)
            assert os.path.exists(path)
            result = np.asarray(PilImage.open(path))
            assert result.shape == (24, 24)
        finally:
            os.unlink(path)

    def test_save_unsupported_format_raises(self):
        """save_image 对不支持的扩展名应抛出 ValueError。"""
        image = np.zeros((10, 10))
        with pytest.raises(ValueError, match='不支持的输出格式'):
            ImageDegradation.save_image('output.bmp', image)


class TestDegradeFromFiles:
    """测试 degrade_from_files 方法。"""

    def _create_test_image_and_psf_files(self, image_shape=(32, 32), psf_shape=(16, 16)):
        """在临时文件中创建图像和 PSF 的 .npy 文件，返回路径元组。"""
        rng = np.random.RandomState(0)
        image = rng.rand(*image_shape)
        psf = rng.rand(*psf_shape) + 0.1

        img_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(img_file.name, image)
        img_file.close()

        psf_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(psf_file.name, psf)
        psf_file.close()

        return img_file.name, psf_file.name

    def test_output_file_created(self):
        """degrade_from_files 处理 .npy 输入后应生成输出文件。"""
        img_path, psf_path = self._create_test_image_and_psf_files()
        try:
            degrader = ImageDegradation()
            _, out_path = degrader.degrade_from_files(img_path, psf_path)
            assert os.path.exists(out_path), f'输出文件应存在：{out_path}'
        finally:
            for p in [img_path, psf_path]:
                os.unlink(p)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_output_shape_correct(self):
        """T_total=1, L_path=0 时输出尺寸应与输入图像一致。"""
        img_path, psf_path = self._create_test_image_and_psf_files(image_shape=(48, 64))
        try:
            degrader = ImageDegradation()
            result, out_path = degrader.degrade_from_files(
                img_path, psf_path, t_total=1.0, l_path=0.0
            )
            assert result.shape == (48, 64), f'输出形状错误：{result.shape}'
        finally:
            for p in [img_path, psf_path]:
                os.unlink(p)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_custom_output_path(self):
        """degrade_from_files 使用指定输出路径时应保存到正确位置。"""
        img_path, psf_path = self._create_test_image_and_psf_files()
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            out_path = f.name
        try:
            degrader = ImageDegradation()
            _, returned_path = degrader.degrade_from_files(
                img_path, psf_path, output_path=out_path
            )
            assert returned_path == out_path
            assert os.path.exists(out_path)
        finally:
            for p in [img_path, psf_path, out_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_default_output_name(self):
        """未指定输出路径时，输出文件名应由输入文件名自动生成。"""
        img_path, psf_path = self._create_test_image_and_psf_files()
        expected_out = img_path.replace('.npy', '_degraded.npy')
        try:
            degrader = ImageDegradation()
            _, out_path = degrader.degrade_from_files(img_path, psf_path)
            assert out_path == expected_out, f'期望输出路径 {expected_out}，实际 {out_path}'
        finally:
            for p in [img_path, psf_path]:
                os.unlink(p)
            if os.path.exists(expected_out):
                os.unlink(expected_out)

    def test_t_total_1_l_path_0_output_shape(self):
        """T_total=1, L_path=0 时返回数组形状正确。"""
        img_path, psf_path = self._create_test_image_and_psf_files(image_shape=(32, 32))
        try:
            degrader = ImageDegradation()
            result, out_path = degrader.degrade_from_files(
                img_path, psf_path, t_total=1.0, l_path=0.0
            )
            assert result.shape == (32, 32)
        finally:
            for p in [img_path, psf_path]:
                os.unlink(p)
            if os.path.exists(out_path):
                os.unlink(out_path)
