import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ncps.torch import CfC
from ncps.wirings import FullyConnected

# ==========================================
# 1. 核心架构
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

def generate_self_recursive(model, initial_X, evolution_ratio, norm_params, warmup_steps=50):
    batch_size, seq_len, _ = initial_X.shape
    device = initial_X.device
    all_outputs = []
    
    hx, buffer, last_mag_raw, last_pred_out = None, None, None, None
    d_min = torch.tensor(norm_params['d_min'][:10], device=device).view(1, 1, 10)
    d_max = torch.tensor(norm_params['d_max'][:10], device=device).view(1, 1, 10)
    m_min = torch.tensor(norm_params['m_min'][:10], device=device).view(1, 1, 10)
    m_max = torch.tensor(norm_params['m_max'][:10], device=device).view(1, 1, 10)

    for t in range(seq_len):
        if t < warmup_steps:
            current_x = initial_X[:, t:t+1, :]
            last_mag_raw = current_x[:, :, 0:10] * (m_max - m_min + 1e-7) + m_min
            last_pred_out = current_x[:, :, 0:10]
        else:
            real_feat = initial_X[:, t:t+1, :].clone()
            mixed_mag_norm = (1 - evolution_ratio) * real_feat[:, :, 0:10] + evolution_ratio * last_pred_out
            mixed_mag_norm = torch.clamp(mixed_mag_norm, min=-0.1, max=1.2) 
            curr_mag_raw = mixed_mag_norm * (m_max - m_min + 1e-7) + m_min
            diff_raw = torch.clamp(curr_mag_raw - last_mag_raw, -(m_max - m_min) * (0.01 if evolution_ratio > 0.8 else 0.05), (m_max - m_min) * (0.01 if evolution_ratio > 0.8 else 0.05))
            curr_mag_raw = last_mag_raw + diff_raw
            
            norm_diff = torch.clamp((diff_raw - d_min) / (d_max - d_min + 1e-7), min=-1.5, max=1.5)
            real_feat[:, :, 0:10] = (curr_mag_raw - m_min) / (m_max - m_min + 1e-7)
            real_feat[:, :, 20:30] = (1 - evolution_ratio) * real_feat[:, :, 20:30] + evolution_ratio * norm_diff
            current_x, last_mag_raw = real_feat, curr_mag_raw
            
        out_step, hx, buffer = model.forward_step(current_x, hx, buffer)
        all_outputs.append(out_step)
        last_pred_out = out_step 

    return torch.cat(all_outputs, dim=1)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = r"D:\ai\models\checkpoint\Evolved_Checkpoint_Ep4900.pth"
    TARGET_PIPE = r"D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv" 
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = AttackStageLNN().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X, Y, pad_len, _ = load_and_preprocess(TARGET_PIPE)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    norm_params = checkpoint['norm_params']
    
    target_ratios = [0.0, 0.8, 1.0]
    results_list = []
    
    for ratio in target_ratios:
        print(f"\n🚀 正在评估 Trans-LNN | Ratio = {ratio}")
        with torch.no_grad():
            student_y = generate_self_recursive(model, X, evolution_ratio=ratio, norm_params=norm_params, warmup_steps=pad_len)

        pred_full = student_y[0, pad_len:, :].cpu().numpy()
        target_full = Y[0, pad_len:, :].cpu().numpy()
        
        # 统计数据计算
        residuals_full = pred_full - target_full
        total_rmse_full = np.sqrt(np.mean(residuals_full**2))
        channel_rmse_full = np.sqrt(np.mean(residuals_full**2, axis=0))
        
        half_idx = int(len(pred_full) * 0.5)
        residuals_half = pred_full[half_idx:, :] - target_full[half_idx:, :]
        total_rmse_half = np.sqrt(np.mean(residuals_half**2))
        
        # 终端直接打印，方便你快速抄写
        print("-" * 40)
        print(f"Total RMSE (Full):    {total_rmse_full:.5f}")
        print(f"Total RMSE (后 50%):  {total_rmse_half:.5f}")
        for i in range(10):
            print(f"  - H{i+1} RMSE: {channel_rmse_full[i]:.5f}")
        print("-" * 40)
        
        # 保存绘图
        plt.figure(figsize=(15, 20))
        plt.suptitle(f"Trans-LNN Manifold Evolution (Ratio = {ratio})\nTotal RMSE: {total_rmse_half:.5f}", fontsize=16)
        for i in range(10): 
            plt.subplot(10, 1, i + 1)
            plt.plot(target_full[:, i], color='#1f77b4', linewidth=1.5, label='Ground Truth')
            plt.plot(pred_full[:, i], color='#d62728', linewidth=1.2, alpha=0.8, label=f'Ratio={ratio}')
            plt.ylabel(f'H {i+1}')
            plt.grid(True, linestyle='--', alpha=0.5)
            if i == 0: plt.legend(loc='upper right')        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        os.makedirs('ablation_results', exist_ok=True)
        plt.savefig(os.path.join('ablation_results', f'Trans_LNN_Ratio_{ratio}.png'), dpi=150)
        plt.close()