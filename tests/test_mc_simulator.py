"""
MCSimulator 单元测试。

验证蒙特卡洛光子追踪仿真器的核心物理正确性。
"""

import numpy as np
import pytest

from atmospheric_mc import AtmosphereParams, MCSimulator


def make_simulator(n_photons=200_000, bins=64, seed=42):
    return MCSimulator(N_photons=n_photons, bins=bins, fov_radius=0.05, seed=seed)


class TestBeerLambert:
    """测试1：Beer-Lambert 弹道透过率验证。

    在纯吸收介质（omega_0=0）中，弹道透过率应满足：
    T_ballistic ≈ exp(-sigma_t * L)

    允许误差 < 5%。
    """

    def test_pure_absorption_thin(self):
        """薄介质（sigma_t*L=0.5）弹道透过率验证。"""
        sigma_t = 0.5
        L = 1.0
        params = AtmosphereParams(
            sigma_t=sigma_t, omega_0=0.0, g=0.75,
            L=L, wavelength_um=10.0, T_atm=280.0, emissivity=0.5,
        )
        sim = make_simulator()
        result = sim.simulate(params)

        T_ball_expected = np.exp(-sigma_t * L)
        T_ball_mc = result['T_ballistic']

        rel_err = abs(T_ball_mc - T_ball_expected) / (T_ball_expected + 1e-30)
        assert rel_err < 0.05, (
            f'弹道透过率误差超标：MC={T_ball_mc:.4f}, '
            f'Beer-Lambert={T_ball_expected:.4f}, 相对误差={rel_err:.2%}'
        )

    def test_pure_absorption_medium(self):
        """中等介质（sigma_t*L=2.0）弹道透过率验证。"""
        sigma_t = 2.0
        L = 1.0
        params = AtmosphereParams(
            sigma_t=sigma_t, omega_0=0.0, g=0.75,
            L=L, wavelength_um=10.0, T_atm=280.0, emissivity=0.5,
        )
        sim = make_simulator()
        result = sim.simulate(params)

        T_ball_expected = np.exp(-sigma_t * L)
        T_ball_mc = result['T_ballistic']

        rel_err = abs(T_ball_mc - T_ball_expected) / (T_ball_expected + 1e-30)
        assert rel_err < 0.05, (
            f'弹道透过率误差超标：MC={T_ball_mc:.4f}, '
            f'Beer-Lambert={T_ball_expected:.4f}, 相对误差={rel_err:.2%}'
        )


class TestPSFNormalization:
    """测试2：PSF 归一化验证。

    PSF 的像素积分应等于 1，误差 < 1%。
    """

    def test_psf_integral_thin_fog(self):
        """薄雾条件下 PSF 归一化验证。"""
        params = AtmosphereParams(
            sigma_t=0.5, omega_0=0.85, g=0.75,
            L=0.5, wavelength_um=10.0, T_atm=275.0, emissivity=0.3,
        )
        sim = make_simulator()
        result = sim.simulate(params)

        psf = result['psf']
        xe = result['xe']
        pixel_area = (xe[1] - xe[0]) ** 2
        integral = psf.sum() * pixel_area

        assert abs(integral - 1.0) < 0.01, (
            f'PSF 归一化误差超标：积分={integral:.4f}，期望 1.0'
        )

    def test_psf_shape(self):
        """验证 PSF 形状正确。"""
        bins = 64
        sim = MCSimulator(N_photons=50_000, bins=bins, fov_radius=0.05, seed=0)
        params = AtmosphereParams(
            sigma_t=1.0, omega_0=0.85, g=0.75,
            L=1.0, wavelength_um=10.0, T_atm=280.0, emissivity=0.5,
        )
        result = sim.simulate(params)

        assert result['psf'].shape == (bins, bins), (
            f'PSF 形状错误：{result["psf"].shape}，期望 ({bins}, {bins})'
        )
        assert result['psf_ballistic'].shape == (bins, bins)
        assert result['psf_scatter'].shape == (bins, bins)

    def test_psf_nonnegative(self):
        """验证 PSF 所有值非负。"""
        params = AtmosphereParams(
            sigma_t=1.0, omega_0=0.85, g=0.75,
            L=0.5, wavelength_um=10.0, T_atm=280.0, emissivity=0.5,
        )
        sim = make_simulator(n_photons=100_000)
        result = sim.simulate(params)

        assert np.all(result['psf'] >= 0), 'PSF 存在负值'


class TestTransmittanceDecomposition:
    """测试3：T_total = T_ballistic + T_scatter 验证。

    总透过率应等于弹道透过率与散射透过率之和，误差 < 1e-4。
    """

    def test_transmittance_decomposition(self):
        """验证透过率分解守恒。"""
        params = AtmosphereParams(
            sigma_t=1.0, omega_0=0.85, g=0.75,
            L=1.0, wavelength_um=10.0, T_atm=280.0, emissivity=0.5,
        )
        sim = make_simulator()
        result = sim.simulate(params)

        T_total = result['T_total']
        T_ball = result['T_ballistic']
        T_sca = result['T_scatter']

        err = abs(T_total - (T_ball + T_sca))
        assert err < 1e-4, (
            f'透过率分解不守恒：T_total={T_total:.6f}, '
            f'T_ball+T_sca={T_ball + T_sca:.6f}, 差={err:.2e}'
        )

    def test_reproducibility_with_seed(self):
        """验证相同 seed 产生可重复结果。"""
        params = AtmosphereParams(
            sigma_t=1.0, omega_0=0.85, g=0.75,
            L=0.5, wavelength_um=10.0, T_atm=280.0, emissivity=0.5,
        )
        sim1 = MCSimulator(N_photons=50_000, bins=64, fov_radius=0.05, seed=123)
        sim2 = MCSimulator(N_photons=50_000, bins=64, fov_radius=0.05, seed=123)

        r1 = sim1.simulate(params)
        r2 = sim2.simulate(params)

        assert r1['T_total'] == r2['T_total'], '相同 seed 结果应完全一致'
        assert np.array_equal(r1['psf'], r2['psf']), '相同 seed PSF 应完全一致'
