import numpy as np
import matplotlib.pyplot as plt

# 參數設定
dx = 0.1
dt = 0.1
x = np.arange(0, 1 + dx, dx)
t = np.arange(0, 1 + dt, dt)  # 模擬到 t = 1

N = len(x)
M = len(t)

# 建立解的陣列 (時間 x 空間)
p = np.zeros((M, N))

# 初始條件
p[0, :] = np.cos(2 * np.pi * x)  # p(x,0)
dp_dt0 = 2 * np.pi * np.sin(2 * np.pi * x)  # ∂p/∂t(x,0)

# 使用 Taylor 展開初始化 p(x,dt)
for i in range(1, N - 1):
    p[1, i] = (p[0, i] + dt * dp_dt0[i] +
               0.5 * (dt**2) * (p[0, i+1] - 2 * p[0, i] + p[0, i-1]) / dx**2)

# 邊界條件
p[:, 0] = 1  # p(0,t) = 1
p[:, -1] = 2  # p(1,t) = 2

# 時間迴圈：由 n=1 開始
for n in range(1, M - 1):
    for i in range(1, N - 1):
        p[n+1, i] = (2 * p[n, i] - p[n-1, i] +
                     (dt/dx)**2 * (p[n, i+1] - 2 * p[n, i] + p[n, i-1]))

# 繪圖
for n in range(0, M, 2):
    plt.plot(x, p[n, :], label=f"t={t[n]:.1f}")

plt.xlabel("x")
plt.ylabel("p(x,t)")
plt.title("Wave Equation Solution")
plt.legend()
plt.grid(True)
plt.show()
# ===== 數值輸出 =====
for time_value in [0.0, 0.5, 1.0]:
    index = int(time_value / dt)
    print(f"\n--- p(x, t={time_value}) ---")
    for i in range(N):
        print(f"x = {x[i]:.1f}, p = {p[index, i]:.6f}")
