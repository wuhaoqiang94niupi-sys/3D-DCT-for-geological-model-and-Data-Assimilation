import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn


class SpectralAnalyzer:
    def __init__(self, data):
        self.data = data
        self.coeffs = None
        self.sorted_indices = None
        self.sorted_magnitudes = None
        self.total_energy = 0

    def perform_dct(self):
        """执行变换并预计算能量"""
        # 使用 ortho 归一化以保证 Parseval 定理 (能量守恒)
        self.coeffs = dctn(self.data, type=2, norm='ortho')
        flat_coeffs = self.coeffs.flatten()

        # 获取排序后的索引 (从大到小)
        # 注意：在实际地质参数化中，我们通常关注'低频'，但在稀疏表达中我们关注'大系数'。
        # DCT的大系数通常就是低频系数。
        self.sorted_indices = np.argsort(np.abs(flat_coeffs))[::-1]
        self.sorted_magnitudes = np.abs(flat_coeffs)[self.sorted_indices]
        self.total_energy = np.sum(self.sorted_magnitudes ** 2)

    def calculate_energy_curve(self):
        """计算能量累积曲线 (对应 Fig. 3a)"""
        cumulative_energy = np.cumsum(self.sorted_magnitudes ** 2)
        energy_ratio = cumulative_energy / self.total_energy
        return energy_ratio

    def partition_spectrum(self, effective_ratio=0.01, num_groups=5):
        """
        实现文稿中的频谱分组策略
        1. 保留 0 号系数 (Mean)
        2. 取前 1% 的系数 (Effective Information)
        3. 将这 1% 分割为 num_groups 个组 (用于后续历史拟合变量 multiplier)
        4. 剩余 99% 标记为 Noise
        """
        n_total = len(self.sorted_magnitudes)
        n_effective = int(n_total * effective_ratio)

        # 确保 Mean (DC分量) 总是被单独处理
        # 在排序后的数组中，DC分量通常是第0个（最大），因为地质场是非负的
        # 但为了严谨，我们显式检查最大值是否为DC，或者简单地按排序分组

        groups = {}

        # Group 0: Mean (通常是最大的系数)
        groups['Mean'] = {
            'indices': self.sorted_indices[0:1],
            'energy_share': (self.sorted_magnitudes[0] ** 2) / self.total_energy
        }

        # Effective Groups (除去 Mean 的前 1%)
        # 将剩余的 n_effective - 1 个系数平均分配给 num_groups
        remainder_effective = n_effective - 1
        group_size = remainder_effective // num_groups

        current_idx = 1
        for i in range(1, num_groups + 1):
            start = current_idx
            end = current_idx + group_size
            # 最后一组吃掉剩下的所有有效系数
            if i == num_groups:
                end = n_effective

            g_indices = self.sorted_indices[start:end]
            g_energy = np.sum(self.sorted_magnitudes[start:end] ** 2)

            groups[f'Group {i}'] = {
                'indices': g_indices,
                'count': len(g_indices),
                'energy_share': g_energy / self.total_energy
            }
            current_idx = end

        # Noise Group (剩余所有)
        noise_indices = self.sorted_indices[n_effective:]
        noise_energy = np.sum(self.sorted_magnitudes[n_effective:] ** 2)
        groups['Noise'] = {
            'indices': noise_indices,
            'count': len(noise_indices),
            'energy_share': noise_energy / self.total_energy
        }

        return groups


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 准备数据
    shape = (50, 36, 134)
    # 使用随机数据模拟，但在真实场景中应加载实际模型
    # 为了让DC分量显著，加上一个基底值
    mock_data = np.random.rand(*shape) * 10 + 50

    analyzer = SpectralAnalyzer(mock_data)
    analyzer.perform_dct()

    # 2. 绘制能量曲线 (Fig. 3a)
    energy_curve = analyzer.calculate_energy_curve()
    x_percent = np.linspace(0, 100, len(energy_curve))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_percent, energy_curve, linewidth=2)
    plt.axvline(x=1.0, color='r', linestyle='--', label='1% Threshold')
    plt.title('Energy Ratio vs % Coefficients (Fig. 3a)')
    plt.xlabel('% of Coefficients')
    plt.ylabel('Cumulative Energy Ratio')
    plt.xlim(-1, 10)  # 放大观察前10%
    plt.ylim(0.8, 1.01)
    plt.grid(True)
    plt.legend()

    # 3. 执行分组策略并可视化 (Fig. 3b / Fig. 6a)
    # 文稿中提到：Mean占52.8%，Group 1占12.3%... [cite: 227, 257]
    groups = analyzer.partition_spectrum(effective_ratio=0.01, num_groups=5)

    group_names = list(groups.keys())
    energy_shares = [g['energy_share'] * 100 for g in groups.values()]  # 转换为百分比

    plt.subplot(1, 2, 2)
    bars = plt.bar(group_names, energy_shares, color=['red'] + ['blue'] * 5 + ['gray'])
    plt.title('Information Content by Group (Fig. 6a)')
    plt.ylabel('Information Percentage (%)')
    plt.xticks(rotation=45)

    # 在柱状图上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("分组详情:")
    for name, info in groups.items():
        count = info.get('count', 1)  # Mean count is 1
        print(f"  {name}: 包含 {count} 个系数, 能量占比 {info['energy_share']:.2%}")