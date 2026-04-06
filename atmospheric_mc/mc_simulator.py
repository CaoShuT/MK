"""
核心蒙特卡洛光子追踪模拟器。

基于权重修正法（Implicit Capture）和 Henyey-Greenstein 相函数，
对大气中的光子传播进行向量化批处理仿真，统计到达探测面的 PSF。
"""

import numpy as np
from .atmosphere import AtmosphereParams


class MCSimulator:
    """
    蒙特卡洛大气光子传播仿真器。

    使用 NumPy 向量化操作批量追踪光子，统计探测面上的
    点扩散函数（PSF）及各透过率分量。

    Args:
        N_photons: 模拟光子数量，默认 2_000_000
        bins: PSF 直方图分辨率（bins x bins），默认 256
        fov_radius: 视场半径 [km]，默认 0.05
        seed: 随机数种子，用于可重复性
    """

    def __init__(
        self,
        N_photons: int = 2_000_000,
        bins: int = 256,
        fov_radius: float = 0.05,
        seed: int = None,
    ):
        self.N_photons = N_photons
        self.bins = bins
        self.fov_radius = fov_radius
        self.seed = seed

    def simulate(self, params: AtmosphereParams) -> dict:
        """
        运行蒙特卡洛光子追踪仿真。

        光子从点源（z=0）出发，沿 +z 方向朝探测面（z=L）传播。
        使用权重修正法处理吸收，HG 相函数采样散射方向，
        俄罗斯轮盘赌截断低权重光子。

        物理模型：
        - 自由程采样：s = -ln(R) / sigma_t  （Beer-Lambert 定律）
        - 权重修正：w *= omega_0  （每次散射后更新权重）
        - 总透过率：T_total = sum(w_hits) / N_photons

        Args:
            params: 大气参数 AtmosphereParams 实例

        Returns:
            dict 包含以下键值：
                'psf':           归一化 PSF，shape (bins, bins)，积分=1
                'psf_ballistic': 弹道光子 PSF 分量
                'psf_scatter':   散射光子 PSF 分量
                'T_total':       总透过率
                'T_ballistic':   弹道透过率 ≈ exp(-sigma_t*L)
                'T_scatter':     散射透过率
                'xe':            PSF x 轴边界数组
                'ye':            PSF y 轴边界数组
                'n_hits':        到达探测面的光子数
        """
        rng = np.random.default_rng(self.seed)
        sigma_t = params.sigma_t
        omega_0 = params.omega_0
        g = params.g
        L = params.L
        N = self.N_photons

        # 初始化光子状态：点源位于 z=0，方向 muz=+1
        x = np.zeros(N, dtype=np.float64)
        y = np.zeros(N, dtype=np.float64)
        z = np.zeros(N, dtype=np.float64)
        mux = np.zeros(N, dtype=np.float64)
        muy = np.zeros(N, dtype=np.float64)
        muz = np.ones(N, dtype=np.float64)   # 朝向探测面
        w = np.ones(N, dtype=np.float64)
        scatter_count = np.zeros(N, dtype=np.int32)

        # 记录击中探测面的光子
        x_hits_list = []
        y_hits_list = []
        w_hits_list = []
        sca_hits_list = []

        alive = np.ones(N, dtype=bool)

        # 主循环：批量推进所有存活光子
        while np.any(alive):
            idx = np.where(alive)[0]
            n_alive = len(idx)

            # 采样自由程：s = -ln(R) / sigma_t
            # rng.random() 返回 (0, 1) 开区间值，无需额外 epsilon
            s = -np.log(rng.random(n_alive)) / sigma_t

            # 记录旧位置
            z_old = z[idx].copy()

            # 更新位置
            x[idx] += s * mux[idx]
            y[idx] += s * muy[idx]
            z[idx] += s * muz[idx]

            # 检测穿越探测面 z=L
            crossed = alive.copy()
            crossed[alive] = z[alive] >= L
            cross_idx = np.where(crossed)[0]

            if len(cross_idx) > 0:
                # 线性插值得精确落点
                z_c_old = z_old[np.isin(idx, cross_idx)] if len(idx) != N else z_old[cross_idx]
                # 获取 cross_idx 在 idx 中的对应旧 z
                mask_in_alive = np.isin(idx, cross_idx)
                z_c_old = z_old[mask_in_alive]
                z_c_new = z[cross_idx]
                muz_c = muz[cross_idx]
                x_c = x[cross_idx]
                y_c = y[cross_idx]
                mux_c = mux[cross_idx]
                muy_c = muy[cross_idx]
                s_c = s[mask_in_alive]

                # 反算到达 z=L 的精确比例 t：z_old + t*dz = L
                dz = z_c_new - z_c_old
                safe_dz = np.abs(dz) > 1e-30
                t_frac = np.where(safe_dz, (L - z_c_old) / dz, 0.0)
                t_frac = np.clip(t_frac, 0.0, 1.0)

                x_hit = x_c - (1.0 - t_frac) * s_c * mux_c
                y_hit = y_c - (1.0 - t_frac) * s_c * muy_c

                x_hits_list.append(x_hit)
                y_hits_list.append(y_hit)
                w_hits_list.append(w[cross_idx].copy())
                sca_hits_list.append(scatter_count[cross_idx].copy())

                alive[cross_idx] = False

            # 丢弃反向传播光子（muz < 0）
            backward = alive & (muz < 0.0)
            alive[backward] = False

            if not np.any(alive):
                break

            # 更新存活光子索引
            alive_idx = np.where(alive)[0]

            # 权重修正散射：w *= omega_0
            w[alive_idx] *= omega_0

            # 散射次数计数
            scatter_count[alive_idx] += 1

            # HG 相函数采样新方向
            new_mux, new_muy, new_muz = self._sample_hg_direction(
                mux[alive_idx], muy[alive_idx], muz[alive_idx], g, rng
            )
            mux[alive_idx] = new_mux
            muy[alive_idx] = new_muy
            muz[alive_idx] = new_muz

            # 俄罗斯轮盘赌截断低权重光子（阈值 1e-3，存活概率 0.1）
            rr_thresh = 1e-3
            rr_survive = 0.1
            low_w = alive & (w < rr_thresh)
            if np.any(low_w):
                low_idx = np.where(low_w)[0]
                survive = rng.random(len(low_idx)) < rr_survive
                w[low_idx[survive]] /= rr_survive
                alive[low_idx[~survive]] = False

        # 汇总击中数据
        if x_hits_list:
            x_hits = np.concatenate(x_hits_list)
            y_hits = np.concatenate(y_hits_list)
            w_hits = np.concatenate(w_hits_list)
            sca_counts = np.concatenate(sca_hits_list)
        else:
            x_hits = np.array([])
            y_hits = np.array([])
            w_hits = np.array([])
            sca_counts = np.array([], dtype=np.int32)

        n_hits = len(x_hits)

        # 构建 PSF 直方图
        R = self.fov_radius
        pixel_area = (2.0 * R / self.bins) ** 2

        hist_range = [[-R, R], [-R, R]]

        psf_raw, xe, ye = np.histogram2d(
            x_hits, y_hits,
            bins=self.bins,
            weights=w_hits,
            range=hist_range,
        )

        # 归一化：积分=1
        total_weight = psf_raw.sum() * pixel_area
        psf = psf_raw / (total_weight + 1e-30)

        # 弹道/散射分量分离
        is_ballistic = sca_counts == 0
        psf_ball_raw, _, _ = np.histogram2d(
            x_hits[is_ballistic] if np.any(is_ballistic) else np.array([]),
            y_hits[is_ballistic] if np.any(is_ballistic) else np.array([]),
            bins=self.bins,
            weights=w_hits[is_ballistic] if np.any(is_ballistic) else np.array([]),
            range=hist_range,
        )
        psf_sca_raw, _, _ = np.histogram2d(
            x_hits[~is_ballistic] if np.any(~is_ballistic) else np.array([]),
            y_hits[~is_ballistic] if np.any(~is_ballistic) else np.array([]),
            bins=self.bins,
            weights=w_hits[~is_ballistic] if np.any(~is_ballistic) else np.array([]),
            range=hist_range,
        )

        psf_ballistic = psf_ball_raw / (total_weight + 1e-30)
        psf_scatter = psf_sca_raw / (total_weight + 1e-30)

        # 透过率计算
        T_total = w_hits.sum() / (N + 1e-30) if n_hits > 0 else 0.0
        w_ball = w_hits[is_ballistic].sum() if np.any(is_ballistic) else 0.0
        w_sca = w_hits[~is_ballistic].sum() if np.any(~is_ballistic) else 0.0
        T_ballistic = w_ball / (N + 1e-30)
        T_scatter = w_sca / (N + 1e-30)

        return {
            'psf': psf,
            'psf_ballistic': psf_ballistic,
            'psf_scatter': psf_scatter,
            'T_total': float(T_total),
            'T_ballistic': float(T_ballistic),
            'T_scatter': float(T_scatter),
            'xe': xe,
            'ye': ye,
            'n_hits': n_hits,
        }

    def _sample_hg_direction(
        self,
        mux: np.ndarray,
        muy: np.ndarray,
        muz: np.ndarray,
        g: float,
        rng: np.random.Generator,
    ):
        """
        Henyey-Greenstein 相函数采样新散射方向。

        公式：
            cos_theta = (1 + g² - ((1-g²)/(1-g+2gR))²) / (2g)  当 |g| >= 1e-6
            cos_theta = 1 - 2R                                    当 |g| < 1e-6（各向同性）

        包含完整旋转变换，处理 gimbal lock（muz≈±1 的情形）。

        Args:
            mux, muy, muz: 当前方向向量分量数组
            g: HG 不对称因子
            rng: NumPy 随机数生成器

        Returns:
            (new_mux, new_muy, new_muz): 归一化后的新方向向量分量
        """
        N = len(mux)
        R = rng.random(N)
        if abs(g) < 1e-6:
            cos_theta = 1.0 - 2.0 * R
        else:
            cos_theta = (
                1.0 + g**2
                - ((1.0 - g**2) / (1.0 - g + 2.0 * g * R + 1e-30)) ** 2
            ) / (2.0 * g + 1e-30)

        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0))
        phi = 2.0 * np.pi * rng.random(N)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        sin_theta0 = np.sqrt(np.maximum(1.0 - muz ** 2, 0.0))
        safe = sin_theta0 > 1e-6

        new_mux = np.where(
            safe,
            (sin_theta * (mux * muz * cos_phi - muy * sin_phi))
            / (sin_theta0 + 1e-30)
            + mux * cos_theta,
            sin_theta * cos_phi,
        )
        new_muy = np.where(
            safe,
            (sin_theta * (muy * muz * cos_phi + mux * sin_phi))
            / (sin_theta0 + 1e-30)
            + muy * cos_theta,
            sin_theta * sin_phi,
        )
        new_muz = np.where(
            safe,
            -sin_theta * sin_theta0 * cos_phi + muz * cos_theta,
            cos_theta * np.sign(muz + 1e-30),
        )

        norm = np.sqrt(new_mux ** 2 + new_muy ** 2 + new_muz ** 2) + 1e-30
        return new_mux / norm, new_muy / norm, new_muz / norm
