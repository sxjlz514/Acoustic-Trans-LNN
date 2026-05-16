import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import CfC
from ncps.wirings import FullyConnected
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
def load_and_preprocess(file_path, ratio=1, downsample=1, zero_pad_len=100):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
        
    df = pd.read_csv(file_path)
    if downsample > 1:
        df = df.iloc[::downsample, :].reset_index(drop=True)
    
    df = df.round(8).astype(np.float32)
    times = df['Time'].values
    dts = np.diff(times, prepend=times[0])
    dts_tensor = torch.tensor(dts, dtype=torch.float32).unsqueeze(1)
    
    # 【核心修复区】：强制分离 Mag 和 Sigma，防止它们交替排列导致切片污染！
    mag_cols = [col for col in df.columns if 'Mag' in col]
    sigma_cols = [col for col in df.columns if 'Sigma' in col]
    
    # 将特征重新拼接为 [Mag1~10, Sigma1~10] 的严格顺序
    raw_mags = df[mag_cols].values
    raw_sigmas = df[sigma_cols].values
    raw_features = np.concatenate([raw_mags, raw_sigmas], axis=1) # Shape: [seq_len, 20]
    
    # 增强输入缩放：给高频特征增加权重
    scale_factors = torch.linspace(1.0, 1.5, 20).numpy()
    
    f_min = raw_features.min(axis=0)
    f_max = raw_features.max(axis=0)
    norm_features = ((raw_features - f_min) / (f_max - f_min + 1e-7)) * scale_factors
    
    raw_diff = np.zeros_like(raw_features)
    raw_diff[1:] = raw_features[1:] - raw_features[:-1]
    raw_diff[0] = 0 # 起始点增量设为0
    
    d_min = raw_diff.min(axis=0)
    d_max = raw_diff.max(axis=0)
    norm_diff = (raw_diff - d_min) / (d_max - d_min + 1e-7) 
    
    weighted_x = torch.tensor(norm_features, dtype=torch.float32)
    weighted_diff = torch.tensor(norm_diff, dtype=torch.float32)
    
    # 2. 严格检查 X_raw_data 的合成
    X_raw_data = torch.cat([weighted_x, weighted_diff, dts_tensor], dim=1)
    
    # 3. 目标 Y 提取
    mag_values = raw_mags
    m_min, m_max = mag_values.min(axis=0), mag_values.max(axis=0)
    Y_raw_data = torch.tensor((mag_values - m_min) / (m_max - m_min + 1e-7), dtype=torch.float32)

    # --- 核心对齐修复：确保没有 Off-by-one 误差 ---
    # 强制同步所有张量的长度，防止 cat 之后出现尾部零填充
    seq_len = min(X_raw_data.size(0), Y_raw_data.size(0))
    X_raw_data = X_raw_data[:seq_len, :]
    Y_raw_data = Y_raw_data[:seq_len, :]

    # 4. Padding 区域处理
    zero_X = torch.zeros((zero_pad_len, 41), dtype=torch.float32)
    # 给 Padding 区间的 dt 赋予平均值，防止 LNN 在初始化阶段 dt 为 0 导致计算崩溃
    zero_X[:, -1] = dts_tensor.mean() 
    zero_Y = torch.zeros((zero_pad_len, 10), dtype=torch.float32)

    X = torch.cat([zero_X, X_raw_data], dim=0).unsqueeze(0)
    Y = torch.cat([zero_Y, Y_raw_data], dim=0).unsqueeze(0)

    # 5. 最终长度校验（打印出来观察，确保两边都是 zero_pad_len + 4000）
    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    norm_params = {'f_min': f_min, 'f_max': f_max, 'd_min': d_min, 'd_max': d_max, 'm_min': m_min, 'm_max': m_max}
    return X, Y, zero_pad_len, times[:seq_len], norm_params

class SotaPureLNN(nn.Module):
    def __init__(self, input_dim=41, output_dim=10, hidden_units=170): 
        super().__init__()
        # 1. 纯血 CfC 架构：输入 41 维直接进入，没有任何注意力或通道预处理
        # 这种 FullyConnected 连线方式是 MIT 原生最常用的 Baseline 形式
        self.wiring = FullyConnected(hidden_units, output_dim)
        # 模式设为 "pure"，禁用内部所有复杂的门控和物理偏移，强制它进行纯数值拟合
        self.ltc = CfC(input_dim, self.wiring, batch_first=True, mode="pure")
        self.fc = nn.Sequential(
            nn.Linear(hidden_units, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [Batch, Seq_len, 41] 包含 Mag, Sigma, Diffs, dt
        # 直接灌入 CfC，没有任何“物理流形”层面的关照
        out, _ = self.ltc(x)
        
        # 输出层映射
        return self.fc(out)

def attack_manifold_loss(pred, target, weight_mask, epoch):
    reconstruction_loss = (F.huber_loss(pred, target, delta=0.5, reduction='none') * weight_mask).mean()
    
    freq_smooth_weights = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]).to(pred.device)
    
    v_pred = pred[:, 1:, :] - pred[:, :-1, :]
    accel_raw = v_pred[:, 1:, :] - v_pred[:, :-1, :]
    v_target = target[:, 1:, :] - target[:, :-1, :]
    
    sharpness_mask = 1.0 / (1.0 + 20.0 * v_target.abs().pow(2))
    
    accel_loss = (accel_raw.pow(2) * freq_smooth_weights.view(1, 1, 10)).mean() 
    smoothness_loss = (v_pred.abs() * sharpness_mask).mean()
    
    total_loss = (reconstruction_loss * 5.0) + (5.0 * accel_loss) + (2.0 * smoothness_loss)
                 
    eps_stationary = 1e-4
    stationary_mask = (v_target.abs() < eps_stationary).float()
    stationary_height_loss = (F.huber_loss(pred[:, :-1, :], target[:, :-1, :], delta=0.5, reduction='none') * stationary_mask).sum() / (stationary_mask.sum() + 1e-6)
    total_loss += 5 * stationary_height_loss
    p_flat = (pred * weight_mask).reshape(-1, 10)
    t_flat = (target * weight_mask).reshape(-1, 10)
    p_norm = F.normalize(p_flat, p=2, dim=0) 
    t_norm = F.normalize(t_flat, p=2, dim=0)
    p_corr = torch.matmul(p_norm.t(), p_norm)
    t_corr = torch.matmul(t_norm.t(), t_norm)
    struct_loss = F.huber_loss(p_corr, t_corr, delta=0.1)
    cos_sim_loss = 1 - F.cosine_similarity(p_flat, t_flat, dim=1).mean()
    coupling_weight = min(0.2, epoch / 200.0)
    total_loss += coupling_weight * (struct_loss * 5.0 + cos_sim_loss * 2.0)
    return total_loss
def train_and_export():
    # 请确保这里的路径正确
    file_path = r'D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv'
    X, Y, pad_len, raw_times, norm_params = load_and_preprocess(file_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = X.to(device), Y.to(device)
    model = SotaPureLNN(input_dim=41, output_dim=10, hidden_units=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
    
    total_len = X.shape[1]
    weight_mask = torch.ones(total_len).to(device).view(1, -1, 1)
    print(f"🚀 启动 A 阶段优化训练 | 设备: {device}")
    model_sota = SotaPureLNN().to(device)
    print(f"📊 SOTA 模型总参数量: {sum(p.numel() for p in model_sota.parameters() if p.requires_grad)}")
    for epoch in range(1500):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        
        if epoch % 100 == 0:
            with torch.no_grad():
                error = (output - Y).abs().mean(dim=2, keepdim=True)
                weight_mask = 1.0 + 10.0 * error 
                weight_mask[:, :pad_len, :] = 0.0 
                weight_mask[:, -2:, :] = 0.0
        loss = attack_manifold_loss(output, Y, weight_mask, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        if (epoch + 1) % 1 == 0: 
            print(f"Epoch [{epoch+1:04d}] | Loss: {loss.item():.5f}")

    # 确保导出目录存在
    save_dir = r"D:\ai\models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "AttackStage_SOTA.pth")
    torch.save({'model_state_dict': model.state_dict(), 'norm_params': norm_params}, save_path)
    print(f"✅ 训练完成并保存至: {save_path}")
    
    print("\n📊 正在生成最终拟合对比图...")

    model.eval()
    with torch.no_grad():
        pred = model(X)[0, pad_len:, :].detach().cpu()
        target = Y[0, pad_len:, :].detach().cpu()

    plt.figure(figsize=(14, 12))

    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.plot(target[:, i], color='black', alpha=0.3, label='Target (Actual)')
        plt.plot(pred[:, i], color='red', linewidth=1.2, label='Pred (Reconstructed)')
        plt.title(f"Harmonic {i+1}")
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_export()