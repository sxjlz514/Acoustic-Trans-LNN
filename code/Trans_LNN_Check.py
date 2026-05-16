import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import random
from ncps.torch import CfC
from ncps.wirings import FullyConnected
def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(514)
# ==========================================
# 1. 核心架构与训练时保持【绝对一致】
# ==========================================
class AttackStageLNN(nn.Module):
    def __init__(self, input_dim=41, output_dim=10, hidden_units=128):
        super().__init__()
        self.num_channels = 20
        self.channel_dim = 16 
        self.channel_projector = nn.Linear(2, self.channel_dim)
        self.layer_norm = nn.LayerNorm(self.channel_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.channel_dim, num_heads=4, batch_first=True)
        self.ltc_input_dim = self.num_channels * self.channel_dim + 1
        self.wiring = FullyConnected(hidden_units, output_dim) 
        self.ltc = CfC(self.ltc_input_dim, self.wiring, batch_first=True, mode="pure")
        self.fc = nn.Sequential(
            nn.Linear(hidden_units, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim)
        )

    def forward_step(self, x_step, hx=None, buffer=None):
        batch_size, _, _ = x_step.shape
        mags_all = x_step[:, :, :20]
        diffs = x_step[:, :, 20:40]
        dt = x_step[:, :, 40:]
        
        curr_mag = mags_all[:, :, :10]
        if buffer is None:
            new_buffer = curr_mag.repeat(1, 15, 1)
            smooth_bias = curr_mag
        else:
            new_buffer = torch.cat([buffer[:, 1:, :], curr_mag], dim=1)
            smooth_bias = new_buffer.mean(dim=1, keepdim=True)
            
        pairs = torch.stack([mags_all, diffs], dim=-1)
        channel_feats = torch.relu(self.channel_projector(pairs))
        channel_feats = self.layer_norm(channel_feats)
        
        attn_in = channel_feats.view(-1, self.num_channels, self.channel_dim)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        combined_feat = attn_out.reshape(batch_size, 1, -1)
        
        ltc_in = torch.cat([combined_feat, dt], dim=-1)
        out, hx_new = self.ltc(ltc_in, hx=hx)
        fc_out = self.fc(out)
        return fc_out + smooth_bias, hx_new, new_buffer

    def forward(self, x):
        pass

# ==========================================
# 2. 原汁原味且包含物理缺陷引擎的递归器
# ==========================================
def apply_physics_defect(feat, t, defect_mode,intensity=1):
    if defect_mode == 'none':
        return feat
    # \mathbf{s}_{leak}(t) = \mathbf{s}_{id}(t) + [-\Delta \sigma, 0]^T, \Delta \sigma = 0.01   
    if defect_mode == 'air_leak':
        sigma_leak_delta = -0.0135 * intensity
        feat[:, :, 10:20] += sigma_leak_delta
        feat[:, :, 30:40] += sigma_leak_delta
        mag_diff_leak_delta = feat[:, :, 0:10] * sigma_leak_delta
        feat[:, :, 20:30] += mag_diff_leak_delta
        feat[:, :, 0:10] += mag_diff_leak_delta
    # \mathbf{s}_{crack}(t) = \mathbf{s}_{id}(t) + [-C_{\sigma} + \xi_{\sigma}, \xi_{\omega}]^T, \xi \sim \mathcal{N}(0, 0.1^2), C_{\sigma} = 0.3   
    elif defect_mode == 'lip_crack':
        crack_penalty = 0.3*intensity
        noise_intensity = 0.1*intensity
        noise = torch.randn_like(feat[:, :, 12:20]) * noise_intensity
        feat[:, :, 12:20] = feat[:, :, 12:20] - crack_penalty + noise
        feat[:, :, 32:40] += torch.randn_like(feat[:, :, 32:40]) * noise_intensity
    # \mathbf{s}_{noise}(t) = \mathbf{s}_{id}(t) + \boldsymbol{\xi}_{base}, \boldsymbol{\xi} \sim \mathcal{N}(0, 0.04^2)   
    elif defect_mode == 'noisy_baseline':
        noise_intensity = 0.06*intensity
        feat[:, :, 10:20] += torch.randn_like(feat[:, :, 10:20]) * noise_intensity
        feat[:, :, 30:40] += torch.randn_like(feat[:, :, 30:40]) * noise_intensity  
    elif defect_mode == 'noisy_baseline_coupled':
        sigma_noise_intensity = 0.073* intensity
        sigma_noise_norm = torch.randn_like(feat[:, :, 10:20]) * sigma_noise_intensity
        feat[:, :, 10:20] += sigma_noise_norm
        feat[:, :, 30:40] += sigma_noise_norm
        mag_diff_noise_norm = feat[:, :, 0:10] * sigma_noise_norm
        feat[:, :, 20:30] += mag_diff_noise_norm
        feat[:, :, 0:10] += mag_diff_noise_norm
    # \delta \mathbf{s}_{flutter}(t) = [A \sin(\Omega t), A \Omega \cos(\Omega t)]^T, A=0.2, \Omega=0.05
    elif defect_mode == 'wind_flutter':
        flutter_freq = 0.05
        flutter_amp = 0.2*intensity
        flutter = math.sin(t * flutter_freq) * flutter_amp
        feat[:, :, 10:20] += flutter
        flutter_diff = math.cos(t * flutter_freq) * flutter_amp * flutter_freq
        feat[:, :, 30:40] += flutter_diff
    # \mathbf{s}_{dust}(t) = [\sigma_{id} - 0.5, \omega_{id} \cdot 0.1]^T
    elif defect_mode == 'dust_accumulation':
        feat[:, :, 14:20] -= 0.5*intensity  
        feat[:, :, 34:40] *=1-(0.9*intensity)
        
    return feat

def generate_TRUE_autonomous_with_defects(model, initial_X, norm_params, warmup_steps=50, defect_mode='none'):
    batch_size, seq_len, _ = initial_X.shape
    device = initial_X.device
    all_outputs = []
    
    hx = None
    buffer = None
    last_mag_raw = None 
    last_pred_out = None
    
    d_min = torch.tensor(norm_params['d_min'][:10], device=device).view(1, 1, 10)
    d_max = torch.tensor(norm_params['d_max'][:10], device=device).view(1, 1, 10)
    m_min = torch.tensor(norm_params['m_min'][:10], device=device).view(1, 1, 10)
    m_max = torch.tensor(norm_params['m_max'][:10], device=device).view(1, 1, 10)

    for t in range(seq_len):
        if t < warmup_steps:
            current_x = initial_X[:, t:t+1, :].clone()
            current_x = apply_physics_defect(current_x, t, defect_mode)
            
            last_mag_raw = current_x[:, :, 0:10] * (m_max - m_min + 1e-7) + m_min
            last_pred_out = current_x[:, :, 0:10]
        else:
            real_feat = initial_X[:, t:t+1, :].clone()
            real_feat = apply_physics_defect(real_feat, t, defect_mode)        
            evolution_ratio = 1.0
            pred_mag_norm = last_pred_out 
            mixed_mag_norm = (1 - evolution_ratio) * real_feat[:, :, 0:10] + evolution_ratio * pred_mag_norm
            mixed_mag_norm = torch.clamp(mixed_mag_norm, min=-0.1, max=1.2) 
            
            curr_mag_raw = mixed_mag_norm * (m_max - m_min + 1e-7) + m_min
            diff_raw = curr_mag_raw - last_mag_raw
            dynamic_jump = 0.01 if evolution_ratio > 0.8 else 0.05
            max_allowed_jump = (m_max - m_min) * dynamic_jump 
            diff_raw = torch.clamp(diff_raw, -max_allowed_jump, max_allowed_jump)
            curr_mag_raw = last_mag_raw + diff_raw
            
            norm_diff = (diff_raw - d_min) / (d_max - d_min + 1e-7)
            norm_diff = torch.clamp(norm_diff, min=-1.5, max=1.5)
            final_mag_norm = (curr_mag_raw - m_min) / (m_max - m_min + 1e-7)
            
            real_feat[:, :, 0:10] = final_mag_norm
            real_feat[:, :, 20:30] = (1 - evolution_ratio) * real_feat[:, :, 20:30] + evolution_ratio * norm_diff
            current_x = real_feat
            last_mag_raw = curr_mag_raw
            
        out_step, hx, buffer = model.forward_step(current_x, hx, buffer)
        all_outputs.append(out_step)
        last_pred_out = out_step 

    return torch.cat(all_outputs, dim=1)

# ==========================================
# 3. 数据加载 (100% 与原版 check2full 保持一致)
# ==========================================
def load_and_preprocess(file_path, downsample=1, zero_pad_len=100):
    df = pd.read_csv(file_path)
    if downsample > 1:
        df = df.iloc[::downsample, :].reset_index(drop=True)
    df = df.round(8).astype(np.float32)
    times = df['Time'].values
    dts = np.diff(times, prepend=times[0])
    dts_tensor = torch.tensor(dts, dtype=torch.float32).unsqueeze(1)
    
    mag_cols = [col for col in df.columns if 'Mag' in col]
    sigma_cols = [col for col in df.columns if 'Sigma' in col]
    raw_mags = df[mag_cols].values
    raw_sigmas = df[sigma_cols].values
    raw_features = np.concatenate([raw_mags, raw_sigmas], axis=1) 
    
    scale_factors = torch.linspace(1.0, 1.5, 20).numpy()
    f_min = raw_features.min(axis=0)
    f_max = raw_features.max(axis=0)
    norm_features = ((raw_features - f_min) / (f_max - f_min + 1e-7)) * scale_factors
    
    raw_diff = np.zeros_like(raw_features)
    raw_diff[1:] = raw_features[1:] - raw_features[:-1]
    raw_diff[0] = 0 
    
    d_min = raw_diff.min(axis=0)
    d_max = raw_diff.max(axis=0)
    norm_diff = (raw_diff - d_min) / (d_max - d_min + 1e-7) 
    
    weighted_x = torch.tensor(norm_features, dtype=torch.float32)
    weighted_diff = torch.tensor(norm_diff, dtype=torch.float32)
    
    X_raw_data = torch.cat([weighted_x, weighted_diff, dts_tensor], dim=1)
    mag_values = raw_mags
    m_min, m_max = mag_values.min(axis=0), mag_values.max(axis=0)
    Y_raw_data = torch.tensor((mag_values - m_min) / (m_max - m_min + 1e-7), dtype=torch.float32)
    
    seq_len = min(X_raw_data.size(0), Y_raw_data.size(0))
    zero_X = torch.zeros((zero_pad_len, 41), dtype=torch.float32)
    zero_X[:, -1] = dts_tensor.mean() 
    zero_Y = torch.zeros((zero_pad_len, 10), dtype=torch.float32)
    
    X = torch.cat([zero_X, X_raw_data[:seq_len, :]], dim=0).unsqueeze(0)
    Y = torch.cat([zero_Y, Y_raw_data[:seq_len, :]], dim=0).unsqueeze(0)
    
    norm_params = {'f_min': f_min, 'f_max': f_max, 'd_min': d_min, 'd_max': d_max, 'm_min': m_min, 'm_max': m_max}
    return X, Y, zero_pad_len, norm_params

# ==========================================
# 4. 残差流形可视化算子
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_residual_perception(real_y, s_out, defect_mode, y_max=0.6):
    """
    计算并绘制绝对能量残差流形，同时量化预测精度
    """
    # 1. 计算原始残差矩阵 (Time, Harmonics)
    if isinstance(real_y, torch.Tensor):
        real_y = real_y.detach().cpu().numpy()
    if isinstance(s_out, torch.Tensor):
        s_out = s_out.detach().cpu().numpy()
        
    # 计算原始残差矩阵
    residuals = s_out - real_y
    if residuals.ndim == 3: # 处理 batch 维度
        residuals = residuals[0]
    
    # 2. 计算各谐波通道的 RMSE 和 全局 RMSE
    # RMSE = sqrt(mean(residual^2))
    channel_rmse = np.sqrt(np.mean(residuals**2, axis=0))
    total_rmse = np.sqrt(np.mean(residuals**2))

 # 3. 开始绘图
    plt.figure(figsize=(12, 8))
    # 使用 viridis 调色盘，H1-H10 层次分明
    colors = plt.cm.viridis(np.linspace(0, 0.9, 10))
    
    for i in range(10):
        # H{i+1} 和 (RMSE...) 之间加入 \n 实现图例内部换行
        plt.plot(residuals[:, i], 
                 color=colors[i], 
                 linewidth=1.5, 
                 alpha=0.8, 
                 label=f'H{i+1}\n(RMSE:{channel_rmse[i]:.4f})')

    # 绘制参考基准，同样加入换行保持格式统一
    plt.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Ideal Healthy\nZero')
    
    # 标题：强调线性比例和固定边界，这是对 -16dB 噪声和 0.01 缺陷的最佳对比视角
    plt.title(f"Dynamic Energy Residual Manifold: {defect_mode.upper()}\n"
              f"(Total RMSE: {total_rmse:.5f} | Isotropic Linear Scale | Fixed Bounds \u00B1{y_max})", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.ylabel('Absolute Energy Deviation ($\delta M$)', fontsize=14)
    plt.xlabel('Time Steps', fontsize=14)
    
    # 核心限制：绝对固定 Y 轴，保持严格的线性比例，杜绝自适应缩放对视觉的误导
    plt.yscale('linear') 
    plt.ylim(-y_max, y_max) 
    
    # 修改这里：去掉了 ncol=2，恢复默认的单列排版，同时保留大字体设置
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Harmonics & Error", 
               frameon=True, fontsize=14, title_fontsize=14)
    
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # 自动创建结果目录
    os.makedirs('inference_results', exist_ok=True)
    save_path = os.path.join('inference_results', f'Residual_Map_{defect_mode}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 50)
    print(f"📊 缺陷模式: {defect_mode.upper()}")
    print(f"📉 全局均方根误差 (Total RMSE): {total_rmse:.6f}")
    print(f"📸 绝对等比例残差流形图像已保存至: {save_path}")
    print("-" * 50)
    
    plt.show()
    
    return total_rmse, channel_rmse
# ==========================================
# 5. 终极验证执行区
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 请确保这里的路径与你本地一致
    MODEL_PATH = r"D:\ai\models\checkpoint\Evolved_Checkpoint_Ep4900.pth"
    TARGET_PIPE = r"D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv" 
    
    # 测试模式选择: 'none', 'air_leak', 'lip_crack', 'wind_flutter', 'noisy_baseline', 'dust_accumulation'
    DEFECT_MODE = 'noisy_baseline_coupled'  # 你可以在这里切换不同的物理缺陷模式进行测试
    
    print(f"📦 正在加载满步演化模型: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    model = AttackStageLNN().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("正在加载管风琴数据...")
    X, Y, pad_len, _ = load_and_preprocess(TARGET_PIPE)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    norm_params = checkpoint['norm_params']
    
    print(f"🚀 启动 100% 纯自主演化脱机推演 | 物理环境: 【{DEFECT_MODE}】")
    with torch.no_grad():
        student_y = generate_TRUE_autonomous_with_defects(
            model, X, norm_params=norm_params, warmup_steps=pad_len, defect_mode=DEFECT_MODE
        )

    # 剥离 Padding 部分，仅保留真实的物理演化时段
    s_out = student_y[0, pad_len:, :].cpu().numpy()
    real_y = Y[0, pad_len:, :].cpu().numpy()
    
    print("绘制高分辨率物理感知残差图...")
    # 直接调用残差流形算子
    plot_residual_perception(real_y, s_out, DEFECT_MODE, y_max=1)