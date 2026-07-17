# DiT & Flow Matching 原理详解与手写实现

> 本文从数学原理出发，循序渐进地讲解 **Diffusion Transformer (DiT)** 和 **Flow Matching**，并提供完整的 PyTorch 手写实现。

---

## 目录

1. [背景：从 DDPM 到 DiT](#背景从-ddpm-到-dit)
2. [DiT 原理详解](#dit-原理)
3. [Flow Matching 原理详解](#flow-matching-原理)
4. [DiT 手写实现](#dit-实现)
5. [Flow Matching 手写实现](#flow-matching-实现)
6. [两者结合：DiT + Flow Matching](#结合dit--flow-matching)
7. [训练与推理完整流程](#完整流程)

---

## 背景：从 DDPM 到 DiT

### DDPM 回顾

DDPM (Ho et al., 2020) 定义了一个**前向加噪过程**：

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\right)
$$

其中 $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$，模型学习**反向去噪**：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)
$$

**核心损失**（简化版）：

$$
\mathcal{L} = \mathbb{E}_{x_0,\ \epsilon,\ t}\left[\ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\ \right]
$$

### 为什么需要 DiT？

| 方法 | Backbone | 特点 / 问题 |
|---|---|---|
| DDPM / DDIM | U-Net (CNN) | 归纳偏置强，扩展性差 |
| **DiT** | **Transformer** | 扩展性极强，遵循 Scaling Law |
| Flow Matching | 任意网络 | 训练更稳定，推理步数少 |

DiT 的核心贡献：**用 Transformer 替换 U-Net 作为去噪网络**，同时提出了高效的条件注入机制。

---

## DiT 原理

### 1. 整体架构

```
输入图像 x ∈ R^(H×W×C)
    ↓ Patchify（把图像切成 patch）
patch tokens ∈ R^(N×d)   [N = HW/p², d = 模型维度]
    ↓ + 位置编码 (sin/cos 或可学习)
    ↓ DiT Block × L
    ↓ unpatchify
去噪图像 x̂
```

### 2. Patchify

将图像 $H \times W \times C$ 切成 $N$ 个 patch，每个 patch 大小为 $p \times p$：

$$
N = \frac{H \cdot W}{p^2}, \qquad \text{token\_dim} = p^2 \cdot C
$$

然后用一个线性层投影到模型维度 $d$。

### 3. 时间步嵌入

时间步 $t$ 通过 **Sinusoidal Embedding + MLP** 编码：

$$
\text{emb}(t) = \text{MLP}\big(\text{SinusoidalEmb}(t)\big) \in \mathbb{R}^d
$$

### 4. 条件注入：adaLN-Zero（核心创新）

DiT 提出了多种条件注入方式，其中 **adaLN-Zero** 效果最好：

**普通 LayerNorm：**

$$
\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

**adaLN（Adaptive LayerNorm）：**

$$
\text{adaLN}(x, c) = \frac{x - \mu}{\sigma} \cdot \big(1 + \gamma(c)\big) + \beta(c)
$$

其中 $\gamma(c), \beta(c)$ 由条件 $c$（时间步 + 类别）通过 MLP 动态生成：

$$
[\gamma, \beta] = \text{Linear}\big(\text{SiLU}(\text{Linear}(c))\big)
$$

**adaLN-Zero** 在此基础上，对残差分支额外引入门控系数 $\alpha(c)$：

$$
x \leftarrow x + \alpha(c) \cdot \text{Attention}\big(\text{adaLN}(x, c)\big)
$$

$$
x \leftarrow x + \alpha(c) \cdot \text{FFN}\big(\text{adaLN}(x, c)\big)
$$

初始化时将 $\alpha$ 对应线性层的参数设为零，使整个残差块在初始时等效于恒等映射，极大地稳定了训练。

### 5. DiT Block 完整结构

```
输入: x（token 序列）, c（条件: t + class）

[adaLN-Zero Block]
1. (shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp) = MLP(SiLU(c))    # 共 6 组参数
2. x_norm = adaLN(x, scale_msa, shift_msa)
3. x = x + gate_msa * Attention(x_norm)
4. x_norm = adaLN(x, scale_mlp, shift_mlp)
5. x = x + gate_mlp * FFN(x_norm)

输出: x
```

### 6. 输出层（Final Layer）

最后一层也用 adaLN：

$$
x = \text{adaLN}(x, c) \ \longrightarrow\ \text{Linear}(x) \ \longrightarrow\ \text{unpatchify}
$$

输出大小为 $N \times (p^2 \cdot C)$，即预测每个 patch 的噪声或 $x_0$。

---

## Flow Matching 原理

### 1. 核心思想

Flow Matching 的目标是学习一个**速度场（velocity field）** $v_\theta(x, t)$，使得从噪声 $x_0 \sim p_0$ 出发，沿着 ODE 流动到数据 $x_1 \sim p_1$：

$$
\frac{dx}{dt} = v_\theta(x_t, t), \qquad t \in [0, 1]
$$

> 注意：Flow Matching 中 $t=0$ 对应噪声，$t=1$ 对应数据（与 DDPM 的时间方向相反）。

### 2. 概率流 ODE

与 DDPM 的 SDE 不同，Flow Matching 使用确定性的 ODE：

$$
dx = v_\theta(x_t, t)\, dt
$$

推理时用数值积分（Euler、RK4 等）求解：

$$
x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t
$$

### 3. Conditional Flow Matching（CFM）目标

**关键问题**：如何定义目标速度场？

对于数据点 $x_1$（真实数据）和噪声 $x_0 \sim \mathcal{N}(0, I)$，定义**条件流**：

$$
x_t = (1 - t)\, x_0 + t\, x_1
$$

对应的**条件速度场**（最优传输路径）：

$$
u_t(x \mid x_0, x_1) = x_1 - x_0
$$

这是一条**直线**——从噪声直接走到数据。

### 4. CFM 损失函数

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\ x_0,\ x_1}\left[\ \lVert v_\theta(x_t, t) - (x_1 - x_0) \rVert^2\ \right]
$$

其中：

- $t \sim \mathcal{U}[0, 1]$（均匀采样时间步）
- $x_0 \sim \mathcal{N}(0, I)$（标准高斯噪声）
- $x_1 \sim p_{\text{data}}$（真实数据）
- $x_t = (1-t)\, x_0 + t\, x_1$（线性插值）

**就这么简单**：目标是让网络预测从 $x_0$ 到 $x_1$ 的方向向量。

### 5. 与 DDPM 的对比

| 特性 | DDPM | Flow Matching |
|---|---|---|
| 前向过程 | 马尔可夫链加噪 | 线性插值 |
| 目标 | 预测噪声 $\epsilon$ | 预测速度 $v = x_1 - x_0$ |
| 路径 | 随机弯曲 | 最优直线（OT） |
| 推理步数 | 数百步 DDPM / ~50 步 DDIM | ~10 步欧拉积分 |
| 训练损失 | 简单 MSE | 简单 MSE |
| 理论框架 | SDE | ODE |

### 6. 为什么 Flow Matching 更好？

1. **直线路径** → 积分更准确，推理步数更少
2. **无马尔可夫假设** → 训练更简单
3. **数学上等价于** Rectified Flow 与 Stochastic Interpolants
4. **Stable Diffusion 3 / FLUX** 都采用了这一方法

---

## DiT 关键结构框图

**整体前向流程（对应 `DiT.forward`）：**

```
输入: x (B,C,H,W) 带噪 latent, t (B,) 时间步, y (B,) 类别(可选)

1. Patchify:        x → (B, N, p²C)          [N = (H/p)·(W/p)]
2. patch_embed:     Linear(p²C → dim)         → (B, N, dim)
3. + pos_embed:     可学习位置编码，逐元素相加
4. 时间嵌入:        SinusoidalEmb(t) → MLP(dim*4→dim) = t_emb
5. 类别嵌入:        y_embedder(y)（可选，+1个无条件类别）
6. c = t_emb (+ y_emb)              # 条件向量 (B, dim)
7. for block in DiT_Blocks:  x = block(x, c)
8. FinalLayer(x, c) → Linear → unpatchify → 输出 (B,C,H,W)
```

**单个 DiT Block（adaLN-Zero）内部结构：**

```
输入: x (B,N,dim), c (B,dim)

c → SiLU → Linear(dim → 6·dim) → 拆分成 6 份:
      shift_msa, scale_msa, gate_msa,
      shift_mlp, scale_mlp, gate_mlp

分支一（注意力）:
   x_norm = LayerNorm(x)（无仿射参数）
   x_norm = x_norm·(1+scale_msa) + shift_msa      # adaLN
   x = x + gate_msa · SelfAttention(x_norm)        # 门控残差

分支二（FFN）:
   x_norm = LayerNorm(x)
   x_norm = x_norm·(1+scale_mlp) + shift_mlp
   x = x + gate_mlp · FFN(x_norm)

要点：Linear(dim→6·dim) 的权重与偏置初始化为 0
      ⇒ 训练开始时 gate=0 ⇒ 整个 Block 恒等映射 ⇒ 训练更稳定
```

**FinalLayer 结构：**

```
c → SiLU → Linear(dim → 2·dim) → shift, scale   (同样零初始化)
x_norm = LayerNorm(x)·(1+scale) + shift
输出 = Linear(x_norm)  → reshape 回 (B,C,H,W)
```

---

## Flow Matching 关键框图

**训练一步（对应 CFM 损失计算）：**

```
输入: x1 (真实数据, B,C,H,W), y(可选)

1. 采样 x0 ~ N(0, I)                  # 与 x1 同形状
2. 采样 t ~ U[0,1]                    # 每个样本独立
3. x_t = (1-t)·x0 + t·x1              # 线性插值
4. v_target = x1 - x0                 # 目标速度（恒定，不随t变）
5. v_pred = 模型(x_t, t, y)
6. loss = MSE(v_pred, v_target)
```

**推理（Euler 积分采样）：**

```
x ~ N(0, I)                     # t=0 起点
dt = 1 / num_steps
for t = 0, dt, 2dt, ..., 1-dt:
    v = 模型(x, t, y)
    x = x + v · dt
返回 x                            # t=1 时即为生成结果
```

**推理（RK4，步数更少精度更高）：**

```
对每个时间步 t，dt：
    k1 = f(x,          t)
    k2 = f(x+dt/2·k1,  t+dt/2)
    k3 = f(x+dt/2·k2,  t+dt/2)
    k4 = f(x+dt·k3,    t+dt)
    x  = x + dt/6 · (k1 + 2k2 + 2k3 + k4)
其中 f(x,t) = 模型预测的速度 v_theta(x,t)
```

**Classifier-Free Guidance（CFG）推理组合：**

```
v_cond   = 模型(x, t, y=类别标签)
v_uncond = 模型(x, t, y=无条件token)
v_guided = v_uncond + cfg_scale · (v_cond - v_uncond)
x = x + v_guided · dt
```

训练时需配合：约 10% 概率把标签替换为"无条件 token"，
使模型同时学会有条件和无条件的速度预测。

---

## 完整流程

### 训练流程图

```
真实图像 x₁
    ↓
采样噪声 x₀ ~ N(0, I)
    ↓
采样时间 t ~ U[0,1]
    ↓
线性插值 x_t = (1-t)x₀ + t·x₁
    ↓
DiT(x_t, t, y) → 预测速度 v̂
    ↓
Loss = ||v̂ - (x₁ - x₀)||²
    ↓
反向传播，更新参数
```

### 推理流程图

```
采样噪声 x₀ ~ N(0, I)   [t=0]
    ↓
for t in [0, dt, 2dt, ..., 1-dt]:
    v = DiT(x_t, t, y)          # 预测速度
    x_(t+dt) = x_t + v · dt     # 欧拉积分
    ↓
x₁   [t=1]   =   生成的图像！
```

### 关键超参数指南

| 参数 | 小型实验 | 生产级 (DiT-XL) |
|---|---|---|
| `dim` | 256 | 1152 |
| `depth` | 6 | 28 |
| `num_heads` | 8 | 16 |
| `patch_size` | 2 | 2 |
| `img_size` (latent) | 32 | 32（对应 256px 图像） |
| `lr` | 1e-4 | 1e-4 |
| `cfg_scale` | 4.0 | 4.0 ~ 7.5 |
| `num_steps` | 30 | 30 ~ 50 |

### 与 Stable Diffusion 3 / FLUX 的联系

- **SD3 / FLUX** = VAE（压缩图像） + DiT + Flow Matching
- 实际上先用 VAE 把图像压缩到 latent space（`in_channels=4`），再在 latent 上训练
- FLUX 还加入了**双流 Transformer（MMDiT）**：文本和图像 token 分开处理，再做交叉注意力

---

## 总结

| 概念 | 一句话总结 |
|---|---|
| **DiT** | 用 Transformer + adaLN-Zero 代替 U-Net 做去噪网络 |
| **adaLN-Zero** | 条件动态生成 LN 参数，零初始化残差门控 |
| **Flow Matching** | 学习从噪声到数据的直线速度场，用 ODE 推理 |
| **CFM 损失** | $\text{MSE}(v_{\text{pred}}, x_1 - x_0)$，极其简单 |
| **CFG** | 条件和无条件速度的加权组合，提升质量 |
| **推理** | 欧拉积分，~30 步即可，比 DDPM 快得多 |