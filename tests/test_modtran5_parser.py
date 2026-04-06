"""
MODTRAN5Parser 单元测试。

使用 io.StringIO 构造模拟文件内容测试解析器功能。
"""

import io
import numpy as np
import pytest

from atmospheric_mc import MODTRAN5Parser


# 模拟 .tp7 文件内容
_TP7_CONTENT = """\
! MODTRAN5 output - test file
! Generated for unit testing
FREQ    TOT_TRANS  PTH_THRML  THRML_SCT  SURF_EMIS  SOL_SCAT  SING_SCAT  GRND_RFLT  DRCT_RFLT  TOTAL_RAD  REF_SOL  SOL@OBS  DEPTH
  900.0  0.800000   0.010000   0.005000   0.200000   0.001000   0.003000   0.000500   0.000100   0.220000   0.000000   0.000000  0.22
  950.0  0.750000   0.015000   0.008000   0.220000   0.002000   0.005000   0.000600   0.000200   0.250000   0.000000   0.000000  0.29
 1000.0  0.600000   0.025000   0.012000   0.250000   0.003000   0.008000   0.000700   0.000300   0.290000   0.000000   0.000000  0.51
 1050.0  0.500000   0.030000   0.015000   0.270000   0.004000   0.010000   0.000800   0.000400   0.310000   0.000000   0.000000  0.69
 1100.0  0.400000   0.040000   0.020000   0.290000   0.005000   0.012000   0.000900   0.000500   0.340000   0.000000   0.000000  0.92
"""

# 模拟 .plt 文件内容
_PLT_CONTENT = """\
! MODTRAN5 plot data - test
! wavenumber  radiance
  800.0  0.010
  900.0  0.020
 1000.0  0.035
 1100.0  0.025
 1200.0  0.015
"""


class TestParseTP7:
    """测试 .tp7 文件解析功能。"""

    def test_basic_parsing(self):
        """基本解析：行数和列数正确。"""
        parser = MODTRAN5Parser()
        df = parser.parse_tp7(io.StringIO(_TP7_CONTENT))

        assert len(df) == 5, f'数据行数错误：{len(df)}，期望 5'
        assert 'FREQ' in df.columns, '缺少 FREQ 列'
        assert 'TOT_TRANS' in df.columns, '缺少 TOT_TRANS 列'

    def test_freq_values(self):
        """验证波数列数值正确解析。"""
        parser = MODTRAN5Parser()
        df = parser.parse_tp7(io.StringIO(_TP7_CONTENT))

        expected_freqs = [900.0, 950.0, 1000.0, 1050.0, 1100.0]
        np.testing.assert_allclose(
            df['FREQ'].values, expected_freqs, rtol=1e-6,
            err_msg='FREQ 列数值解析错误',
        )

    def test_transmittance_values(self):
        """验证透过率列数值正确。"""
        parser = MODTRAN5Parser()
        df = parser.parse_tp7(io.StringIO(_TP7_CONTENT))

        expected_trans = [0.8, 0.75, 0.6, 0.5, 0.4]
        np.testing.assert_allclose(
            df['TOT_TRANS'].values, expected_trans, rtol=1e-6,
            err_msg='TOT_TRANS 列数值解析错误',
        )

    def test_skip_comments(self):
        """验证注释行被正确跳过。"""
        content = "! 这是注释\n! 这也是注释\n  1000.0  0.5  0.01  0.005  0.02  0.001  0.003  0.0005  0.0001  0.03  0.0  0.0  0.69\n"
        parser = MODTRAN5Parser()
        df = parser.parse_tp7(io.StringIO(content))
        assert len(df) == 1, f'应该只有 1 行数据，实际 {len(df)} 行'


class TestWavenumberWavelengthConversion:
    """测试波数-波长转换。"""

    def test_10um_to_wavenumber(self):
        """10.0 μm 应转换为 1000 cm⁻¹。"""
        wavelength_um = 10.0
        expected_wn = 10000.0 / wavelength_um
        assert abs(expected_wn - 1000.0) < 1e-6, (
            f'波数转换错误：{expected_wn} cm⁻¹，期望 1000 cm⁻¹'
        )

    def test_extract_mc_params_wavenumber(self):
        """extract_mc_params 应在正确波数附近查找数据。"""
        parser = MODTRAN5Parser()
        df = parser.parse_tp7(io.StringIO(_TP7_CONTENT))

        # 10 μm → 1000 cm⁻¹，最近行是 1000.0
        params = parser.extract_mc_params(df, wavelength_um=10.0, path_length_km=1.0)

        assert abs(params['wavenumber_used'] - 1000.0) < 1.0, (
            f'最近波数错误：{params["wavenumber_used"]}，期望接近 1000.0'
        )
        assert abs(params['T_total'] - 0.6) < 0.01, (
            f'T_total 提取错误：{params["T_total"]}，期望 0.6'
        )

    def test_extract_mc_params_sigma_t(self):
        """验证 sigma_t 从 TOT_TRANS 正确计算。"""
        parser = MODTRAN5Parser()
        df = parser.parse_tp7(io.StringIO(_TP7_CONTENT))
        # 1000 cm⁻¹ 处 TOT_TRANS = 0.6，L=1 km
        # sigma_t = -ln(0.6) / 1.0 ≈ 0.5108
        params = parser.extract_mc_params(df, wavelength_um=10.0, path_length_km=1.0)
        expected_sigma_t = -np.log(0.6) / 1.0
        assert abs(params['sigma_t'] - expected_sigma_t) < 0.01, (
            f'sigma_t 计算错误：{params["sigma_t"]:.4f}，期望 {expected_sigma_t:.4f}'
        )


class TestParsePlt:
    """测试 .plt 文件解析功能。"""

    def test_basic_parsing(self):
        """基本解析：行数和列名正确。"""
        parser = MODTRAN5Parser()
        df = parser.parse_plt(io.StringIO(_PLT_CONTENT))

        assert len(df) == 5, f'数据行数错误：{len(df)}，期望 5'
        assert list(df.columns) == ['wavenumber', 'radiance'], (
            f'列名错误：{list(df.columns)}'
        )

    def test_values(self):
        """验证解析的数值正确。"""
        parser = MODTRAN5Parser()
        df = parser.parse_plt(io.StringIO(_PLT_CONTENT))

        expected_wn = [800.0, 900.0, 1000.0, 1100.0, 1200.0]
        np.testing.assert_allclose(
            df['wavenumber'].values, expected_wn, rtol=1e-6
        )

    def test_get_path_radiance_interpolation(self):
        """验证路径辐射插值功能。"""
        parser = MODTRAN5Parser()
        df = parser.parse_plt(io.StringIO(_PLT_CONTENT))

        # 1000 cm⁻¹ 对应 10 μm，辐亮度应为 0.035
        rad = parser.get_path_radiance(df, wavelength_um=10.0)
        assert abs(rad - 0.035) < 0.001, (
            f'路径辐射插值错误：{rad:.4f}，期望 0.035'
        )

    def test_empty_input(self):
        """空文件应返回空 DataFrame。"""
        parser = MODTRAN5Parser()
        df = parser.parse_plt(io.StringIO("! only comments\n"))
        assert df.empty or len(df) == 0
