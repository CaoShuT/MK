"""
大气参数数据类与云雾预设工厂函数。

定义描述大气传输效应所需的物理参数，并提供常见云雾场景的预设值。
"""

from dataclasses import dataclass


@dataclass
class AtmosphereParams:
    """
    大气传输参数数据类。

    Attributes:
        sigma_t: 消光系数 [km⁻¹]，sigma_t = sigma_abs + sigma_sca
        omega_0: 单次散射反照率，omega_0 = sigma_sca / sigma_t，取值 [0, 1]
        g: Henyey-Greenstein 不对称因子，取值 (-1, 1)，正值为前向散射
        L: 大气路径长度 [km]
        wavelength_um: 工作波长 [μm]
        T_atm: 大气等效温度 [K]，用于计算路径辐射
        emissivity: 大气有效发射率，取值 [0, 1]
    """
    sigma_t: float        # 消光系数 [km⁻¹]
    omega_0: float        # 单次散射反照率
    g: float              # HG 不对称因子
    L: float              # 大气路径长度 [km]
    wavelength_um: float  # 工作波长 [μm]
    T_atm: float          # 大气等效温度 [K]
    emissivity: float     # 大气有效发射率


def thin_fog(wavelength_um: float, L: float) -> AtmosphereParams:
    """
    薄雾预设参数工厂函数。

    典型薄雾条件：能见度约 2-5 km，轻度散射，红外透过率较高。

    Args:
        wavelength_um: 工作波长 [μm]
        L: 大气路径长度 [km]

    Returns:
        AtmosphereParams: 薄雾大气参数实例
    """
    return AtmosphereParams(
        sigma_t=0.5,
        omega_0=0.85,
        g=0.75,
        L=L,
        wavelength_um=wavelength_um,
        T_atm=275.0,
        emissivity=0.3,
    )


def medium_fog(wavelength_um: float, L: float) -> AtmosphereParams:
    """
    中雾预设参数工厂函数。

    典型中雾条件：能见度约 0.5-2 km，中等散射，红外透过率明显降低。

    Args:
        wavelength_um: 工作波长 [μm]
        L: 大气路径长度 [km]

    Returns:
        AtmosphereParams: 中雾大气参数实例
    """
    return AtmosphereParams(
        sigma_t=2.0,
        omega_0=0.9,
        g=0.80,
        L=L,
        wavelength_um=wavelength_um,
        T_atm=278.0,
        emissivity=0.6,
    )


def thick_fog(wavelength_um: float, L: float) -> AtmosphereParams:
    """
    浓雾/云预设参数工厂函数。

    典型浓雾/云条件：能见度小于 0.5 km，强散射，红外透过率极低。

    Args:
        wavelength_um: 工作波长 [μm]
        L: 大气路径长度 [km]

    Returns:
        AtmosphereParams: 浓雾大气参数实例
    """
    return AtmosphereParams(
        sigma_t=8.0,
        omega_0=0.92,
        g=0.85,
        L=L,
        wavelength_um=wavelength_um,
        T_atm=280.0,
        emissivity=0.9,
    )
