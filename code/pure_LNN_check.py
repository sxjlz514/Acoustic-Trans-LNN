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
# 1. SOTA 架构 (纯净连续时间，无 Attention)
# ==========================================
class SotaPureLNN(nn.Module):
    def __init__(self, input_dim=41, output_dim=10, hidden_units=128): 
        super().__init__()
        self.wiring = FullyConnected(hidden_units, output_dim)
        self.ltc = CfC(input_dim, self.wiring, batch_first=True, mode="pure")
        self.fc = nn.Sequential(
            nn.Linear(hidden_units, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward_step(self, x_step, hx=None, buffer=None):
        out, hx_new = self.ltc(x_step, hx=hx)
        fc_out = self.fc(out)
        return fc_out, hx_new, None 

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
        if torch.isnan(out_step).any():
            out_step = torch.nan_to_num(out_step, nan=0.0)
        all_outputs.append(out_step)
        last_pred_out = out_step 

    return torch.cat(all_outputs, dim=1)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 确信这里加载的是你 SOTA 隔离训练出的权重
    MODEL_PATH = r"D:\ai\models\checkpoint_sota\SOTA_Best_Evolved_Model.pth"
    TARGET_PIPE = r"D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv" 
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = SotaPureLNN().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X, Y, pad_len, _ = load_and_preprocess(TARGET_PIPE)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    norm_params = checkpoint['norm_params']
    
    target_ratios = [0.0, 0.8, 1.0]
    
    for ratio in target_ratios:
        print(f"\n🚀 正在评估 SOTA 基准模型 | Ratio = {ratio}")
        with torch.no_grad():
            student_y = generate_self_recursive(model, X, evolution_ratio=ratio, norm_params=norm_params, warmup_steps=pad_len)

        pred_full = student_y[0, pad_len:, :].cpu().numpy()
        target_full = Y[0, pad_len:, :].cpu().numpy()
        
        residuals_full = pred_full - target_full
        total_rmse_full = np.sqrt(np.mean(residuals_full**2))
        channel_rmse_full = np.sqrt(np.mean(residuals_full**2, axis=0))
        
        half_idx = int(len(pred_full) * 0.5)
        residuals_half = pred_full[half_idx:, :] - target_full[half_idx:, :]
        total_rmse_half = np.sqrt(np.mean(residuals_half**2))
        
        print("-" * 40)
        print(f"Total RMSE (Full):    {total_rmse_full:.5f}")
        print(f"Total RMSE (后 50%):  {total_rmse_half:.5f}")
        for i in range(10):
            print(f"  - H{i+1} RMSE: {channel_rmse_full[i]:.5f}")
        print("-" * 40)
        
        plt.figure(figsize=(15, 20))
        plt.suptitle(f"SOTA LNN Evolution (Ratio = {ratio})\nTotal RMSE: {total_rmse_half:.5f}", fontsize=16)
        for i in range(10): 
            plt.subplot(10, 1, i + 1)
            plt.plot(target_full[:, i], color='#1f77b4', linewidth=1.5, label='Ground Truth')
            plt.plot(pred_full[:, i], color='#ff7f0e', linewidth=1.2, alpha=0.8, label=f'SOTA Ratio={ratio}')
            plt.ylabel(f'H {i+1}')
            plt.grid(True, linestyle='--', alpha=0.5)
            if i == 0: plt.legend(loc='upper right')        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        os.makedirs('ablation_results', exist_ok=True)
        plt.savefig(os.path.join('ablation_results', f'SOTA_LNN_Ratio_{ratio}.png'), dpi=150)
        plt.close()