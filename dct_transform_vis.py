import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter


def generate_synthetic_reservoir(shape):
    """
    生成一个具有地质特征（层状/非均质）的合成渗透率场
    而非简单的随机噪声，以便更好地展示DCT的压缩能力。
    """
    # 生成随机噪声并进行高斯平滑，模拟地质连续性
    noise = np.random.randn(*shape)
    # sigma=(2, 10, 10) 模拟层状结构 (Z方向变化快，XY方向变化慢)
    log_perm = gaussian_filter(noise, sigma=(2, 8, 8))
    # 转换到渗透率域 (Log-Normal分布)
    perm = np.exp(log_perm) * 100
    return perm


def process_reservoir_dct(perm_field, keep_ratio=0.01):
    """
    执行完整的 DCT 处理流程：Log变换 -> DCT -> 截断 -> IDCT -> Exp变换
    对应文稿中保留 1% 系数的操作 [cite: 64, 149]。
    """
    # 1. Log 变换 (处理渗透率的非高斯分布特性)
    log_data = np.log1p(perm_field)

    # 2. 3D-DCT 正变换
    coeffs = dctn(log_data, type=2, norm='ortho')

    # 3. 频域截断 (模拟压缩)
    flat_coeffs = coeffs.flatten()
    # 按绝对值排序找到阈值
    threshold_idx = int(len(flat_coeffs) * keep_ratio)
    sorted_abs = np.sort(np.abs(flat_coeffs))[::-1]
    threshold_val = sorted_abs[threshold_idx]

    # 将低于阈值的系数置零 (Hard Thresholding)
    coeffs_truncated = np.where(np.abs(coeffs) >= threshold_val, coeffs, 0)

    # 4. 3D-DCT 逆变换
    log_recon = idctn(coeffs_truncated, type=2, norm='ortho')

    # 5. 反向 Log 变换
    perm_recon = np.expm1(log_recon)

    return coeffs, perm_recon


def plot_comparison(original, reconstructed, coeffs):
    """可视化对比：空间域切片 vs 频域切片"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 选取中间切片 (Z方向)
    z_slice = original.shape[2] // 2

    # 1. 原始渗透率场
    im1 = axes[0].imshow(original[:, :, z_slice], cmap='jet', aspect='auto')
    axes[0].set_title(f'Original Permeability (Slice Z={z_slice})')
    plt.colorbar(im1, ax=axes[0], label='mD')

    # 2. 重构渗透率场 (1% 系数)
    im2 = axes[1].imshow(reconstructed[:, :, z_slice], cmap='jet', aspect='auto')
    axes[1].set_title('Reconstructed (Top 1% Coeffs)')
    plt.colorbar(im2, ax=axes[1], label='mD')

    # 3. 频域能量分布 (Log Scale)
    # 展示 DCT 系数的 Log 幅度，验证能量是否集中在左上角 (低频区)
    # 取第一层切片观察
    spectrum_slice = np.log10(np.abs(coeffs[:, :, 0]) + 1e-6)
    im3 = axes[2].imshow(spectrum_slice, cmap='inferno', aspect='auto')
    axes[2].set_title('DCT Spectrum (Log Magnitude)')
    axes[2].set_xlabel('Fy (Frequency Y)')
    axes[2].set_ylabel('Fx (Frequency X)')
    plt.colorbar(im3, ax=axes[2], label='Log10(Magnitude)')

    plt.tight_layout()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 设置模型尺寸 (参考文稿 Table 2: 50x36x134)
    grid_shape = (50, 36, 134)
    print(f"生成合成油藏模型 {grid_shape}...")

    real_perm = generate_synthetic_reservoir(grid_shape)

    # 2. 处理与压缩
    print("执行 3D-DCT 及 1% 压缩...")
    full_coeffs, recon_perm = process_reservoir_dct(real_perm, keep_ratio=0.01)

    # 3. 计算误差
    rel_error = np.linalg.norm(real_perm - recon_perm) / np.linalg.norm(real_perm)
    print(f"保留 1% 系数后的相对误差: {rel_error:.4%}")
    print("注：误差主要来自高频细节丢失，但主要地质趋势应被保留 [cite: 11]。")

    # 4. 绘图
    plot_comparison(real_perm, recon_perm, full_coeffs)