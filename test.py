import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm

# Try multiple Chinese fonts as fallbacks
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 生成随机二维矩阵 (16x16，符合小波变换要求)
np.random.seed(42)
input_matrix = np.random.rand(16, 16)
print("输入矩阵形状:", input_matrix.shape)

# 2. 应用二维小波变换 (DWT)
# 使用Daubechies小波基 (db1)
coeffs = pywt.dwt2(input_matrix, 'db1')
print(coeffs)
LL, (LH, HL, HH) = coeffs

# 3. 拼接四个子带形成完整系数矩阵 (16x16)
# 注意：每个子带尺寸为输入的一半 (8x8)
# 拼接方式: [LL, LH; HL, HH]
coeff_matrix = np.zeros((16, 16))
coeff_matrix[:8, :8] = LL  # 低频
coeff_matrix[:8, 8:] = LH  # 水平高频
coeff_matrix[8:, :8] = HL  # 垂直高频
coeff_matrix[8:, 8:] = HH  # 对角高频

print("小波变换后系数矩阵形状:", coeff_matrix.shape)

# 4. 可视化输入和输出
plt.figure(figsize=(12, 5))

# Create font properties for Chinese text
font_prop = fm.FontProperties(family='sans-serif', size=10)

# 输入图像
plt.subplot(1, 2, 1)
plt.imshow(input_matrix, cmap='viridis')
plt.title('Original Input Matrix (16x16)', fontsize=12)
plt.colorbar()

# 小波系数矩阵
plt.subplot(1, 2, 2)
plt.imshow(coeff_matrix, cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1))
plt.title('Wavelet Coefficients Matrix (16x16)', fontsize=12)
plt.colorbar()

plt.tight_layout()
plt.savefig('wavelet_transform.svg')
plt.show()

# 5. 输出系数矩阵到文件 (可选)
np.save('wavelet_coeffs.npy', coeff_matrix)
print("\n小波系数矩阵已保存至 'wavelet_coeffs.npy'")
print("系数矩阵示例 (前3行3列):")
print(coeff_matrix[:3, :3])