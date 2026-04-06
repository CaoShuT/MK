"""
MODTRAN5 参数优化器。

利用 scipy 优化算法，通过最小化蒙特卡洛仿真结果与
MODTRAN5 输出之间的差异，对大气模型参数进行校准。
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from .atmosphere import AtmosphereParams


class MODTRANOptimizer:
    """
    基于 MODTRAN5 输出的大气参数优化器。

    通过最小化蒙特卡洛仿真透过率与 MODTRAN5 参考值之间的差异，
    优化 Henyey-Greenstein 不对称因子 g 及其他大气参数。
    """

    def optimize_g(
        self,
        sigma_t: float,
        omega_0: float,
        L: float,
        T_scatter_target: float,
        mc_simulator,
        g_bounds: tuple = (0.5, 0.99),
        n_photons_opt: int = 300_000,
    ) -> float:
        """
        优化 HG 不对称因子 g，使散射透过率匹配 MODTRAN5 参考值。

        使用 scipy.optimize.minimize_scalar（'bounded' 方法）。
        损失函数：(T_scatter_MC - T_scatter_target)²

        Args:
            sigma_t: 消光系数 [km⁻¹]
            omega_0: 单次散射反照率
            L: 大气路径长度 [km]
            T_scatter_target: MODTRAN5 参考散射透过率
            mc_simulator: MCSimulator 实例（simulate 方法会被临时修改 N_photons）
            g_bounds: g 的搜索边界，默认 (0.5, 0.99)
            n_photons_opt: 优化阶段使用的光子数，默认 300_000

        Returns:
            优化得到的最优 g 值
        """
        # 保存原始光子数，优化时使用较小光子数以提高速度
        original_n = mc_simulator.N_photons
        mc_simulator.N_photons = n_photons_opt

        call_count = [0]

        def loss(g_val):
            call_count[0] += 1
            params = AtmosphereParams(
                sigma_t=sigma_t,
                omega_0=omega_0,
                g=float(g_val),
                L=L,
                wavelength_um=10.0,
                T_atm=280.0,
                emissivity=0.6,
            )
            result = mc_simulator.simulate(params)
            T_scatter_mc = result['T_scatter']
            loss_val = (T_scatter_mc - T_scatter_target) ** 2
            print(
                f'  [optimize_g] iter={call_count[0]:3d}  g={g_val:.4f}  '
                f'T_scatter_MC={T_scatter_mc:.6f}  target={T_scatter_target:.6f}  '
                f'loss={loss_val:.2e}'
            )
            return loss_val

        result = minimize_scalar(loss, bounds=g_bounds, method='bounded')
        mc_simulator.N_photons = original_n

        print(f'[optimize_g] 最优 g = {result.x:.4f}，最终损失 = {result.fun:.2e}')
        return float(result.x)

    def optimize_all(
        self,
        modtran_params: dict,
        L: float,
        wavelength_um: float,
        mc_simulator,
        sigma_t_tol: float = 0.2,
        omega_0_tol: float = 0.05,
        n_photons_opt: int = 300_000,
    ) -> dict:
        """
        同时优化 sigma_t、omega_0 和 g 三个参数。

        使用 scipy.optimize.minimize（L-BFGS-B 方法）。
        损失函数：10*(T_total_err)² + (T_scatter_err)²

        边界设置：
        - sigma_t：在 MODTRAN5 值基础上 ±20%
        - omega_0：在 MODTRAN5 值基础上 ±0.05，截断到 [0, 0.999]
        - g：[0.5, 0.99]

        Args:
            modtran_params: extract_mc_params() 返回的参数字典，
                            包含 'sigma_t', 'omega_0', 'T_total', 'T_scatter'
            L: 大气路径长度 [km]
            wavelength_um: 工作波长 [μm]
            mc_simulator: MCSimulator 实例
            sigma_t_tol: sigma_t 的相对容差（默认 ±20%）
            omega_0_tol: omega_0 的绝对容差（默认 ±0.05）
            n_photons_opt: 优化阶段光子数，默认 300_000

        Returns:
            dict 包含：
                'sigma_t': 优化后消光系数
                'omega_0': 优化后单次散射反照率
                'g':       优化后不对称因子
                'loss':    最终损失值
        """
        original_n = mc_simulator.N_photons
        mc_simulator.N_photons = n_photons_opt

        T_total_ref = modtran_params['T_total']
        T_scatter_ref = modtran_params['T_scatter']
        sig0 = modtran_params['sigma_t']
        om0 = modtran_params['omega_0']

        # 初始值和边界
        x0 = [sig0, om0, 0.85]
        bounds = [
            (sig0 * (1 - sigma_t_tol), sig0 * (1 + sigma_t_tol)),
            (max(0.0, om0 - omega_0_tol), min(0.999, om0 + omega_0_tol)),
            (0.5, 0.99),
        ]

        call_count = [0]

        def loss(x):
            call_count[0] += 1
            sigma_t_val, omega_0_val, g_val = x
            params = AtmosphereParams(
                sigma_t=float(sigma_t_val),
                omega_0=float(omega_0_val),
                g=float(g_val),
                L=L,
                wavelength_um=wavelength_um,
                T_atm=280.0,
                emissivity=0.6,
            )
            result = mc_simulator.simulate(params)
            T_total_mc = result['T_total']
            T_scatter_mc = result['T_scatter']
            loss_val = (
                10.0 * (T_total_mc - T_total_ref) ** 2
                + (T_scatter_mc - T_scatter_ref) ** 2
            )
            print(
                f'  [optimize_all] iter={call_count[0]:3d}  '
                f'σ_t={sigma_t_val:.3f} ω₀={omega_0_val:.3f} g={g_val:.3f}  '
                f'T_tot={T_total_mc:.4f}(ref={T_total_ref:.4f})  '
                f'T_sca={T_scatter_mc:.4f}(ref={T_scatter_ref:.4f})  '
                f'loss={loss_val:.2e}'
            )
            return loss_val

        opt_result = minimize(
            loss, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-8},
        )
        mc_simulator.N_photons = original_n

        sigma_t_opt, omega_0_opt, g_opt = opt_result.x
        print(
            f'[optimize_all] 完成：σ_t={sigma_t_opt:.4f}  '
            f'ω₀={omega_0_opt:.4f}  g={g_opt:.4f}  '
            f'loss={opt_result.fun:.2e}'
        )

        return {
            'sigma_t': float(sigma_t_opt),
            'omega_0': float(omega_0_opt),
            'g': float(g_opt),
            'loss': float(opt_result.fun),
        }

    @staticmethod
    def validate(
        mc_result: dict,
        modtran_params: dict,
        tol: float = 0.02,
    ) -> dict:
        """
        对比蒙特卡洛仿真结果与 MODTRAN5 参考值，验证参数准确性。

        Args:
            mc_result: MCSimulator.simulate() 返回的结果字典
            modtran_params: extract_mc_params() 返回的参数字典
            tol: 透过率误差容限（绝对值），默认 0.02

        Returns:
            dict 包含：
                'T_total_err':   T_total 绝对误差
                'T_scatter_err': T_scatter 绝对误差
                'passed':        是否两项误差均在 tol 内
        """
        T_total_mc = mc_result['T_total']
        T_scatter_mc = mc_result['T_scatter']
        T_total_ref = modtran_params.get('T_total', 0.5)
        T_scatter_ref = modtran_params.get('T_scatter', 0.0)

        T_total_err = abs(T_total_mc - T_total_ref)
        T_scatter_err = abs(T_scatter_mc - T_scatter_ref)
        passed = (T_total_err <= tol) and (T_scatter_err <= tol)

        print('\n[validate] 验证结果对比：')
        print(f'  {'指标':<15} {'MC仿真':>12} {'MODTRAN5参考':>12} {'绝对误差':>10}')
        print(f'  {"-"*52}')
        print(f'  {'T_total':<15} {T_total_mc:>12.6f} {T_total_ref:>12.6f} {T_total_err:>10.6f}')
        print(f'  {'T_scatter':<15} {T_scatter_mc:>12.6f} {T_scatter_ref:>12.6f} {T_scatter_err:>10.6f}')
        print(f'  验证{"通过 ✓" if passed else "失败 ✗"}（容限 ±{tol}）')

        return {
            'T_total_err': float(T_total_err),
            'T_scatter_err': float(T_scatter_err),
            'passed': bool(passed),
        }
