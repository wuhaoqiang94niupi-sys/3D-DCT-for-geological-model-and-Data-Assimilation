import numpy as np
from scipy.fft import dctn, idctn


class DCTParameterization:
    def __init__(self, data_shape, n_components=12):
        """
        初始化参数化类
        n_components: 需要用于历史拟合的变量数量 (Nvar)，文稿中示例选了12个 [cite: 227]
        """
        self.shape = data_shape
        self.n_components = n_components
        self.basis_indices = None  # 存储被选中的系数索引
        self.mean_val = None
        self.noise_field = None  # 存储剩下的高频噪声部分
        self.basis_vectors = []  # 存储基向量

    def decompose(self, prior_model):
        """
        对先验模型进行分解，提取基向量和噪声 (对应文稿 Methodology 部分)
        """
        # 1. 3D-DCT 变换
        coeffs = dctn(prior_model, type=2, norm='ortho')

        # 2. 提取均值 (DC分量, index 0,0,0)
        self.mean_val = coeffs[0, 0, 0]

        # 3. 找到能量最大的前 N 个系数 (排除均值)
        # 对应文稿: Selecting regions by spectrums / effective coefficients [cite: 151]
        flat_coeffs = coeffs.flatten()
        # 将均值位置设为0，以免重复选择
        flat_coeffs[0] = 0

        # 获取绝对值最大的 n_components 个系数的索引
        top_indices = np.argsort(np.abs(flat_coeffs))[::-1][:self.n_components]
        self.basis_indices = np.unravel_index(top_indices, self.shape)

        # 4. 构建基向量 (Basis Vectors)
        # 对应文稿 Eq. 10 - 12，每个基向量对应一个特定频率的余弦波
        self.basis_vectors = []
        for i in range(self.n_components):
            # 创建一个只有该系数为1，其余为0的稀疏系数矩阵
            sparse_coeff = np.zeros(self.shape)
            idx = tuple(ind[i] for ind in self.basis_indices)
            sparse_coeff[idx] = coeffs[idx]  # 保留原始振幅 magnitude

            # 逆变换得到基向量在空间域的形态
            basis_img = idctn(sparse_coeff, type=2, norm='ortho')
            self.basis_vectors.append(basis_img)

        # 5. 计算噪声部分 (Noise)
        # Noise = Original - Mean - Sum(Basis)
        # 对应文稿 Eq. 16 中的 m_noise
        reconstructed_basis = sum(self.basis_vectors)
        # 注意：均值部分也要逆变换回去 (是一个常数场)
        mean_field = np.full(self.shape, self.mean_val) / np.sqrt(np.prod(self.shape))  # 简化的理解，实际通过逆变换更准

        # 更严谨的计算噪声方法：将选中的系数和均值系数置0，然后逆变换
        noise_coeffs = coeffs.copy()
        noise_coeffs[0, 0, 0] = 0
        for i in range(self.n_components):
            idx = tuple(ind[i] for ind in self.basis_indices)
            noise_coeffs[idx] = 0
        self.noise_field = idctn(noise_coeffs, type=2, norm='ortho')

        # 存储均值场用于重构
        mean_coeffs = np.zeros(self.shape)
        mean_coeffs[0, 0, 0] = self.mean_val
        self.mean_field_spatial = idctn(mean_coeffs, type=2, norm='ortho')

        print(f"分解完成: 提取了 {self.n_components} 个主要特征分量。")

    def reconstruction(self, multipliers):
        """
        根据更新的乘数 w 重构地质模型 (对应文稿公式 16)
        Args:
            multipliers: list or array of size n_components.
                         对应文稿中的 w1, w2... [cite: 189]
                         初始值为 1.0
        Returns:
            new_model: 更新后的 3D 渗透率场
        """
        if len(multipliers) != self.n_components:
            raise ValueError(f"需要 {self.n_components} 个乘数，但提供了 {len(multipliers)} 个")

        # Formula: m_HM = m_mean + sum(w_i * Basis_i) + m_noise
        # 对应文稿 Eq. 16 [cite: 188]
        new_model = self.mean_field_spatial + self.noise_field

        for w, basis in zip(multipliers, self.basis_vectors):
            new_model += w * basis

        return new_model


# --- 模拟历史拟合过程中的模型更新 ---
if __name__ == "__main__":
    # 1. 模拟一个先验渗透率模型
    grid_size = (50, 36, 134)
    prior_perm = np.exp(np.random.randn(*grid_size))  # 对数正态分布模拟

    # 2. 初始化参数化工具
    # 文稿中选择了12个分量作为初始尝试 [cite: 227]
    dct_param = DCTParameterization(grid_size, n_components=12)

    # 3. 分解模型
    dct_param.decompose(prior_model=prior_perm)

    # 4. 生成新模型 (模拟 ES-MDA 更新 w)
    # 初始状态 w 都是 1.0
    initial_w = np.ones(12)
    # 假设 ES-MDA 计算出新的 multipliers，比如增加了第2个分量的权重，减少了第5个
    updated_w = np.ones(12)
    updated_w[1] = 1.5  # 增强某个低频特征
    updated_w[4] = 0.8  # 减弱某个特征

    posterior_perm = dct_param.reconstruction(updated_w)

    print("模型更新完成。")
    print(f"原始模型均值: {np.mean(prior_perm):.4f}")
    print(f"更新后模型均值: {np.mean(posterior_perm):.4f}")
    # 均值变化通常很小，因为均值系数(DC)没有被 w 调整，
    # 变化仅来自于不同基向量的加权组合带来的局部变化。