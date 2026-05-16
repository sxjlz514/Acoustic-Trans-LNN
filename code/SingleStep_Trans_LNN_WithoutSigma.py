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

# 设置线程数
torch.set_num_threads(8)
def attack_manifold_loss_detailed(pred, target, weight_mask, epoch):
    # 1. 重建损失 (Reconstruction) - 衡量数值对齐
    recon_loss = (F.huber_loss(pred, target, delta=0.5, reduction='none') * weight_mask).mean()
    
    # 2. 物理加速度与平滑约束 (Physics Constraints)
    freq_smooth_weights = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]).to(pred.device)
    v_pred = pred[:, 1:, :] - pred[:, :-1, :]
    accel_raw = v_pred[:, 1:, :] - v_pred[:, :-1, :]
    v_target = target[:, 1:, :] - target[:, :-1, :]
    sharpness_mask = 1.0 / (1.0 + 20.0 * v_target.abs().pow(2))
    
    accel_loss = (accel_raw.pow(2) * freq_smooth_weights.view(1, 1, 10)).mean() 
    smooth_loss = (v_pred.abs() * sharpness_mask).mean()
    
    # 3. 稳态高度约束 (Stationary Height)
    eps_stationary = 1e-4
    stationary_mask = (v_target.abs() < eps_stationary).float()
    height_loss = (F.huber_loss(pred[:, :-1, :], target[:, :-1, :], delta=0.5, reduction='none') * stationary_mask).sum() / (stationary_mask.sum() + 1e-6)
    
    # 4. 结构一致性 (Structural/Correlation)
    p_flat = (pred * weight_mask).reshape(-1, 10)
    t_flat = (target * weight_mask).reshape(-1, 10)
    p_norm = F.normalize(p_flat, p=2, dim=0) 
    t_norm = F.normalize(t_flat, p=2, dim=0)
    p_corr = torch.matmul(p_norm.t(), p_norm)
    t_corr = torch.matmul(t_norm.t(), t_norm)
    struct_loss = F.huber_loss(p_corr, t_corr, delta=0.1)
    
    # 动态权重：前 200 个 Epoch 逐渐增加结构耦合
    coupling_weight = min(0.2, epoch / 200.0)
    
    # 计算总损失
    total_loss = (recon_loss * 5.0) + (5.0 * accel_loss) + (2.0 * smooth_loss) + (5 * height_loss) + (coupling_weight * struct_loss * 5.0)
                 
    return total_loss, recon_loss, accel_loss, smooth_loss, height_loss, struct_loss

# ==========================================
# 1. 阉割版数据加载 (彻底移除 Sigma 导数项)
# ==========================================
def load_and_preprocess_degenerate(file_path, downsample=1, zero_pad_len=100):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
        
    df = pd.read_csv(file_path)
    if downsample > 1:
        df = df.iloc[::downsample, :].reset_index(drop=True)
    
    df = df.round(8).astype(np.float32)
    times = df['Time'].values
    dts = np.diff(times, prepend=times[0])
    dts_tensor = torch.tensor(dts, dtype=torch.float32).unsqueeze(1)
    
    # 【核心阉割点】：只提取 Mag 列，完全忽略 Sigma
    mag_cols = [col for col in df.columns if 'Mag' in col]
    raw_mags = df[mag_cols].values # Shape: [seq_len, 10]
    
    # 输入特征仅保留 Mag (10维)
    raw_features = raw_mags 
    
    f_min, f_max = raw_features.min(axis=0), raw_features.max(axis=0)
    norm_features = (raw_features - f_min) / (f_max - f_min + 1e-7)
    
    # 差分项也仅针对 Mag
    raw_diff = np.zeros_like(raw_features)
    raw_diff[1:] = raw_features[1:] - raw_features[:-1]
    d_min, d_max = raw_diff.min(axis=0), raw_diff.max(axis=0)
    norm_diff = (raw_diff - d_min) / (d_max - d_min + 1e-7) 
    
    weighted_x = torch.tensor(norm_features, dtype=torch.float32)
    weighted_diff = torch.tensor(norm_diff, dtype=torch.float32)
    
    # X 维度: 10(Mag) + 10(Diff) + 1(dt) = 21 维
    X_raw_data = torch.cat([weighted_x, weighted_diff, dts_tensor], dim=1)
    
    m_min, m_max = raw_mags.min(axis=0), raw_mags.max(axis=0)
    Y_raw_data = torch.tensor((raw_mags - m_min) / (m_max - m_min + 1e-7), dtype=torch.float32)

    seq_len = min(X_raw_data.size(0), Y_raw_data.size(0))
    X_raw_data, Y_raw_data = X_raw_data[:seq_len, :], Y_raw_data[:seq_len, :]

    # Padding 逻辑保持一致，X 的输入维度改为 21
    zero_X = torch.zeros((zero_pad_len, 21), dtype=torch.float32)
    zero_X[:, -1] = dts_tensor.mean() 
    zero_Y = torch.zeros((zero_pad_len, 10), dtype=torch.float32)

    X = torch.cat([zero_X, X_raw_data], dim=0).unsqueeze(0)
    Y = torch.cat([zero_Y, Y_raw_data], dim=0).unsqueeze(0)

    norm_params = {'f_min': f_min, 'f_max': f_max, 'd_min': d_min, 'd_max': d_max, 'm_min': m_min, 'm_max': m_max}
    return X, Y, zero_pad_len, times[:seq_len], norm_params

# ==========================================
# 2. 残疾版模型 (增加 Hidden Units 以对齐参数量)
# ==========================================
class AblationNoSigmaLNN(nn.Module):
    def __init__(self, input_dim=21, output_dim=10, hidden_units=224): # 增加宽度确保总参数 > 65k
        super().__init__()
        self.num_channels = 10
        self.channel_dim = 32 # 增加通道维度
        
        self.channel_projector = nn.Linear(2, self.channel_dim)
        self.layer_norm = nn.LayerNorm(self.channel_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.channel_dim, num_heads=4, batch_first=True)
        
        self.ltc_input_dim = self.num_channels * self.channel_dim + 1
        self.wiring = FullyConnected(hidden_units, output_dim) 
        self.ltc = CfC(self.ltc_input_dim, self.wiring, batch_first=True, mode="pure")
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_units, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        mags = x[:, :, :10] 
        diffs = x[:, :, 10:20]
        dt = x[:, :, 20:]
        
        # 平滑基准线逻辑（保持一致）
        mags_to_pool = mags.transpose(1, 2)
        mags_padded = F.pad(mags_to_pool, (7, 7), mode='replicate') 
        smooth_bias = F.avg_pool1d(mags_padded, kernel_size=15, stride=1).transpose(1, 2)
        
        pairs = torch.stack([mags, diffs], dim=-1)
        channel_feats = torch.relu(self.channel_projector(pairs))
        channel_feats = self.layer_norm(channel_feats)
        attn_in = channel_feats.view(-1, self.num_channels, self.channel_dim)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        combined_feat = attn_out.reshape(batch_size, seq_len, -1)
        
        ltc_in = torch.cat([combined_feat, dt], dim=-1)
        out, _ = self.ltc(ltc_in)
        fc_out = self.fc(out)
        
        return fc_out + smooth_bias

# ==========================================
# 3. 训练与评估
# ==========================================
def train_ablation_no_sigma():
    file_path = r'D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv'
    X, Y, pad_len, raw_times, norm_params = load_and_preprocess_degenerate(file_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y = X.to(device), Y.to(device)
    
    # 2. 初始化残疾版模型 (对齐 65k 参数)
    model = AblationNoSigmaLNN(input_dim=21, output_dim=10, hidden_units=224).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 50)
    print(f"📊 [架构对齐检查] 阉割版模型 (AblationNoSigmaLNN)")
    print(f"🔢 总可训练参数量: {total_params:,}")
    print("=" * 50)
    # 新增：Loss 历史记录器
    history = {'total': [], 'recon': [], 'accel': [], 'smooth': [], 'height': [], 'struct': []}
    
    total_len = X.shape[1]
    weight_mask = torch.ones(total_len).to(device).view(1, -1, 1)
    
    print(f"🚀 启动 50 Epochs 诊断性训练 | 设备: {device}")
    
    # 修改训练次数为 50 次
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        
        # 掩码更新逻辑
        if epoch % 10 == 0:
            with torch.no_grad():
                error = (output - Y).abs().mean(dim=2, keepdim=True)
                weight_mask = 1.0 + 10.0 * error 
                weight_mask[:, :pad_len, :] = 0.0 
                weight_mask[:, -2:, :] = 0.0
        
        # 获取分解后的损失项
        res = attack_manifold_loss_detailed(output, Y, weight_mask, epoch)
        total_l, recon_l, accel_l, smooth_l, height_l, struct_l = res
        
        # 记录数据
        history['total'].append(total_l.item())
        history['recon'].append(recon_l.item())
        history['accel'].append(accel_l.item())
        history['smooth'].append(smooth_l.item())
        history['height'].append(height_l.item())
        history['struct'].append(struct_l.item())
        
        total_l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        # 实时打印所有关键项，观察“打架”过程
        print(f"E[{epoch+1:02d}] Tot:{total_l:.4f} | Rec:{recon_l:.5f} | Acc:{accel_l:.5f} | Str:{struct_l:.5f}")
    save_dir = r"D:\ai\models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 保存消融版单步模型
    save_path = os.path.join(save_dir, "AttackStage_Ablation_SingleStep_50.pth")
    torch.save({
        'epoch': 50,
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'history': history # 顺便保存 Loss 历史，方便之后写论文直接调用数据
    }, save_path)
    
    print(f"✅ 诊断性单步模型已保存至: {save_path}")
    # --- 绘制 Loss 分解趋势图 ---
    plt.figure(figsize=(12, 7))
    eps = range(1, 51)
    
    plt.plot(eps, history['total'], 'k-o', label='Total (Weighted Sum)', linewidth=2)
    plt.plot(eps, history['recon'], 'r--', label='Reconstruction (Huber)')
    plt.plot(eps, history['accel'], 'g-', label='Acceleration (Physics)')
    plt.plot(eps, history['height'], 'b-.', label='Stationary Height')
    plt.plot(eps, history['struct'], 'm:', label='Structural Consistency')
    
    plt.yscale('log') # 使用对数坐标观察细节
    plt.title("Detailed Loss Analysis (Ablation Model - First 50 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value (Log Scale)")
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()

    # --- 绘制拟合对比图 ---
    print("\n📊 正在生成残疾版拟合对比图...")
    model.eval()
    with torch.no_grad():
        pred = model(X)[0, pad_len:, :].detach().cpu()
        target = Y[0, pad_len:, :].detach().cpu()

    plt.figure(figsize=(14, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.plot(target[:, i], color='black', alpha=0.3, label='Target')
        plt.plot(pred[:, i], color='blue', label='Pred (No Sigma)')
        plt.title(f"Harmonic {i+1}")
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_ablation_no_sigma()