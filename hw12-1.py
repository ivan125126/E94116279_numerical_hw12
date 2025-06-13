import numpy as np

# 網格參數
pi = np.pi
h = k = 0.1 * pi
Nx = int(pi / h)  # x 方向內部點數
Ny = int((pi / 2) / k)  # y 方向內部點數

# 建立網格
x = np.linspace(0, pi, Nx + 1)
y = np.linspace(0, pi/2, Ny + 1)

# 初始化解矩陣 u
u = np.zeros((Nx + 1, Ny + 1))

# 設定邊界條件
# 左右邊界：u(0,y) = cos(y), u(pi, y) = -cos(y)
for j in range(Ny + 1):
    u[0, j] = np.cos(y[j])
    u[Nx, j] = -np.cos(y[j])

# 上下邊界：u(x,0) = cos(x), u(x,pi/2) = 0
for i in range(Nx + 1):
    u[i, 0] = np.cos(x[i])
    u[i, Ny] = 0

# 疊代求解內部點（使用 Gauss-Seidel 方法）
max_iter = 10000
tolerance = 1e-6

for iteration in range(max_iter):
    max_error = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            rhs = h**2 * x[i] * y[j]
            u_new = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - rhs)
            error = abs(u_new - u[i, j])
            max_error = max(max_error, error)
            u[i, j] = u_new
    if max_error < tolerance:
        print(f'Converged in {iteration+1} iterations.')
        break

# 輸出結果
print("u(x,y) approximate values:")
for j in range(Ny + 1):
    for i in range(Nx + 1):
        print(f"u({x[i]:.2f}, {y[j]:.2f}) = {u[i,j]:.6f}")
