"""
MODTRAN5 输出文件解析器。

支持解析 MODTRAN5 标准输出格式（.tp7, .tp6, .plt），
并提取用于蒙特卡洛仿真的大气参数。
"""

import io
import numpy as np
import pandas as pd


# MODTRAN5 .tp7 文件的标准列名
_TP7_COLUMNS = [
    'FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT',
    'SURF_EMIS', 'SOL_SCAT', 'SING_SCAT', 'GRND_RFLT',
    'DRCT_RFLT', 'TOTAL_RAD', 'REF_SOL', 'SOL@OBS', 'DEPTH',
]


class MODTRAN5Parser:
    """
    MODTRAN5 输出文件解析工具。

    支持解析以下文件格式：
    - .tp7：传输通道辐射计算输出
    - .tp6（TAPE6）：逐层大气参数详细报告
    - .plt：绘图数据文件（两列：波数 + 辐亮度）
    """

    @staticmethod
    def parse_tp7(filepath) -> pd.DataFrame:
        """
        解析 MODTRAN5 .tp7 输出文件。

        .tp7 文件为固定列宽文本格式，典型列包括：
        FREQ, TOT_TRANS, PTH_THRML, THRML_SCT, SURF_EMIS,
        SOL_SCAT, SING_SCAT, GRND_RFLT, DRCT_RFLT, TOTAL_RAD,
        REF_SOL, SOL@OBS, DEPTH

        解析规则：
        - 跳过以 '!' 开头的注释行
        - 自动检测包含 'FREQ' 的列头行
        - 若未找到标准列头，使用通用列名 col_0, col_1, ...
        - 跳过无法解析为数值的行

        Args:
            filepath: .tp7 文件路径或类文件对象（如 io.StringIO）

        Returns:
            解析得到的 DataFrame，列名对应 MODTRAN5 输出字段
        """
        if isinstance(filepath, (str,)):
            with open(filepath, 'r') as f:
                lines = f.readlines()
        else:
            # 类文件对象（如 StringIO）
            lines = filepath.readlines()
            if lines and isinstance(lines[0], bytes):
                lines = [ln.decode('utf-8') for ln in lines]

        columns = None
        data_rows = []

        for line in lines:
            stripped = line.strip()
            # 跳过注释行
            if stripped.startswith('!'):
                continue
            # 检测列头行
            if columns is None and 'FREQ' in stripped.upper():
                columns = stripped.split()
                continue
            # 解析数据行
            if stripped == '':
                continue
            parts = stripped.split()
            try:
                row = [float(p) for p in parts]
                data_rows.append(row)
            except ValueError:
                continue

        if not data_rows:
            return pd.DataFrame()

        # 确定列名
        n_cols = len(data_rows[0])
        if columns is None:
            columns = [f'col_{i}' for i in range(n_cols)]
        else:
            # 对齐列数
            if len(columns) < n_cols:
                columns = columns + [f'col_{i}' for i in range(len(columns), n_cols)]
            elif len(columns) > n_cols:
                columns = columns[:n_cols]

        # 过滤列数不一致的行
        data_rows = [r for r in data_rows if len(r) == n_cols]

        return pd.DataFrame(data_rows, columns=columns)

    @staticmethod
    def parse_tp6_layers(filepath) -> pd.DataFrame:
        """
        从 MODTRAN5 TAPE6 详细报告中提取逐层大气参数。

        搜索包含 'COMPUT' 或 'LAYER' 关键字的数据块，提取：
        ALT（高度）、EXT（消光系数）、SCA（散射系数）、
        ABS（吸收系数）、OMEGA_0（单次散射反照率）

        Args:
            filepath: .tp6 文件路径或类文件对象

        Returns:
            包含逐层参数的 DataFrame，列名为
            ['ALT', 'EXT', 'SCA', 'ABS', 'OMEGA_0']；
            若解析失败则返回空 DataFrame
        """
        if isinstance(filepath, str):
            with open(filepath, 'r') as f:
                lines = f.readlines()
        else:
            lines = filepath.readlines()
            if lines and isinstance(lines[0], bytes):
                lines = [ln.decode('utf-8') for ln in lines]

        data_rows = []
        header_found = False
        col_names = ['ALT', 'EXT', 'SCA', 'ABS', 'OMEGA_0']

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('!'):
                continue
            upper = stripped.upper()
            # 寻找数据块头部
            if not header_found and ('COMPUT' in upper or 'LAYER' in upper):
                header_found = True
                continue
            if header_found:
                parts = stripped.split()
                if len(parts) >= 5:
                    try:
                        row = [float(p) for p in parts[:5]]
                        data_rows.append(row)
                    except ValueError:
                        continue

        if not data_rows:
            return pd.DataFrame()

        return pd.DataFrame(data_rows, columns=col_names)

    @staticmethod
    def parse_plt(filepath) -> pd.DataFrame:
        """
        解析 MODTRAN5 .plt 绘图数据文件。

        .plt 文件为两列文本格式：波数 [cm⁻¹] + 辐亮度值。
        跳过以 '!' 开头的注释行，解析所有包含两列数值的行。

        Args:
            filepath: .plt 文件路径或类文件对象

        Returns:
            包含两列的 DataFrame，列名为 ['wavenumber', 'radiance']
        """
        if isinstance(filepath, str):
            with open(filepath, 'r') as f:
                lines = f.readlines()
        else:
            lines = filepath.readlines()
            if lines and isinstance(lines[0], bytes):
                lines = [ln.decode('utf-8') for ln in lines]

        data_rows = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('!') or stripped == '':
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                row = [float(parts[0]), float(parts[1])]
                data_rows.append(row)
            except ValueError:
                continue

        if not data_rows:
            return pd.DataFrame(columns=['wavenumber', 'radiance'])

        return pd.DataFrame(data_rows, columns=['wavenumber', 'radiance'])

    @staticmethod
    def extract_mc_params(
        tp7_df: pd.DataFrame,
        wavelength_um: float,
        path_length_km: float,
    ) -> dict:
        """
        从解析的 .tp7 DataFrame 中提取蒙特卡洛仿真参数。

        波长-波数转换：wavenumber = 10000.0 / wavelength_um  [cm⁻¹]

        提取规则：
        - 在 FREQ 列中查找最接近目标波数的行
        - TOT_TRANS → sigma_t = -ln(TOT_TRANS + 1e-30) / path_length_km
        - SING_SCAT（若存在）→ 估算 omega_0；否则默认 omega_0 = 0.9

        Args:
            tp7_df: parse_tp7() 返回的 DataFrame
            wavelength_um: 目标工作波长 [μm]
            path_length_km: 大气路径长度 [km]

        Returns:
            dict 包含：
                'sigma_t':       消光系数 [km⁻¹]
                'omega_0':       单次散射反照率
                'T_total':       总透过率（来自 TOT_TRANS）
                'T_scatter':     散射透过率（来自 THRML_SCT + SOL_SCAT，若存在）
                'wavenumber_used': 实际使用的波数 [cm⁻¹]
        """
        if wavelength_um <= 0:
            raise ValueError(f'wavelength_um 必须大于 0，实际值: {wavelength_um}')

        wavenumber = 10000.0 / wavelength_um

        if tp7_df.empty:
            return {
                'sigma_t': 1.0, 'omega_0': 0.9,
                'T_total': 0.5, 'T_scatter': 0.1,
                'wavenumber_used': wavenumber,
            }

        # 获取 FREQ 列（第一列）
        freq_col = tp7_df.columns[0]
        freq_vals = tp7_df[freq_col].values
        idx = int(np.argmin(np.abs(freq_vals - wavenumber)))
        row = tp7_df.iloc[idx]
        wavenumber_used = float(freq_vals[idx])

        # 提取总透过率
        T_total = 1.0
        for col in ['TOT_TRANS', 'col_1']:
            if col in tp7_df.columns:
                T_total = float(row[col])
                break

        # 计算消光系数
        sigma_t = -np.log(T_total + 1e-30) / (path_length_km + 1e-30)

        # 估算单次散射反照率
        omega_0 = 0.9
        if 'SING_SCAT' in tp7_df.columns:
            sing_scat = float(row['SING_SCAT'])
            total_rad = 1.0
            if 'TOTAL_RAD' in tp7_df.columns:
                total_rad = float(row['TOTAL_RAD'])
            if total_rad > 1e-10:
                omega_0 = min(0.999, max(0.0, sing_scat / (total_rad + 1e-30)))

        # 散射透过率估算
        T_scatter = 0.0
        for col in ['THRML_SCT', 'SOL_SCAT']:
            if col in tp7_df.columns:
                T_scatter += float(row[col])

        return {
            'sigma_t': float(sigma_t),
            'omega_0': float(omega_0),
            'T_total': float(T_total),
            'T_scatter': float(T_scatter),
            'wavenumber_used': float(wavenumber_used),
        }

    @staticmethod
    def get_path_radiance(plt_df: pd.DataFrame, wavelength_um: float) -> float:
        """
        从 .plt 数据中插值获取指定波长的路径辐射值。

        将波长转换为波数后，在 plt_df 中进行线性插值。

        Args:
            plt_df: parse_plt() 返回的 DataFrame（wavenumber, radiance）
            wavelength_um: 目标波长 [μm]

        Returns:
            插值得到的路径辐射值；若数据不足则返回 0.0
        """
        if plt_df.empty or len(plt_df) < 2:
            return 0.0

        if wavelength_um <= 0:
            raise ValueError(f'wavelength_um 必须大于 0，实际值: {wavelength_um}')

        wavenumber = 10000.0 / wavelength_um
        wn = plt_df['wavenumber'].values
        rad = plt_df['radiance'].values

        # 线性插值
        return float(np.interp(wavenumber, wn, rad))
