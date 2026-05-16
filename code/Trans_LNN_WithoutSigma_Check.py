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
# 1. 残疾模型架构 (严格匹配 canjiTea.py)
# ==========================================
class AblationNoSigmaLNN(nn.Module):
    def __init__(self, input_dim=21, output_dim=10, hidden_units=224):
        super().__init__()
        self.num_channels = 10
        self.channel_dim = 32 
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

    def forward_step(self, x_step, hx=None, buffer=None):
        batch_size, _, _ = x_step.shape
        mags = x_step[:, :, :10] 
        diffs = x_step[:, :, 10:20]
        dt = x_step[:, :, 20:]
        
        if buffer is None:
            new_buffer = mags.repeat(1, 15, 1)
            smooth_bias = mags
        else:
            new_buffer = torch.cat([buffer[:, 1:, :], mags], dim=1)
            smooth_bias = new_buffer.mean(dim=1, keepdim=True)
            
        pairs = torch.stack([mags, diffs], dim=-1)
        channel_feats = torch.relu(self.channel_projector(pairs))
        channel_feats = self.layer_norm(channel_feats)
        attn_in = channel_feats.view(-1, self.num_channels, self.channel_dim)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        combined_feat = attn_out.reshape(batch_size, 1, -1)
        
        ltc_in = torch.cat([combined_feat, dt], dim=-1)
        out, hx_new = self.ltc(ltc_in, hx=hx)
        fc_out = self.fc(out)
        return fc_out + smooth_bias, hx_new, new_buffer

# ==========================================
# 2. 残疾版递归引擎 (支持 Ratio 调节)
# ==========================================
def generate_self_recursive_degenerate(model, initial_X, evolution_ratio, norm_params, warmup_steps=50):
    batch_size, seq_len, _ = initial_X.shape
    device = initial_X.device
    all_outputs = []
    hx, buffer, last_mag_raw, last_pred_out = None, None, None, None
    
    m_min = torch.tensor(norm_params['m_min'], device=device).view(1, 1, 10)
    m_max = torch.tensor(norm_params['m_max'], device=device).view(1, 1, 10)
    d_min = torch.tensor(norm_params['d_min'], device=device).view(1, 1, 10)
    d_max = torch.tensor(norm_params['d_max'], device=device).view(1, 1, 10)

    for t in range(seq_len):
        if t < warmup_steps:
            current_x = initial_X[:, t:t+1, :]
            last_mag_raw = current_x[:, :, 0:10] * (m_max - m_min + 1e-7) + m_min
            last_pred_out = current_x[:, :, 0:10]
        else:
            real_feat = initial_X[:, t:t+1, :].clone()
            pred_mag_norm = last_pred_out 
            
            # 混合逻辑
            mixed_mag_norm = (1 - evolution_ratio) * real_feat[:, :, 0:10] + evolution_ratio * pred_mag_norm
            curr_mag_raw = torch.clamp(mixed_mag_norm, -0.1, 1.2) * (m_max - m_min + 1e-7) + m_min
            diff_raw = curr_mag_raw - last_mag_raw
            
            # 物理限制 (这里的松紧直接决定了残疾模型崩坏的速度)
            max_j = (m_max - m_min) * (0.01 if evolution_ratio > 0.8 else 0.05)
            diff_raw = torch.clamp(diff_raw, -max_j, max_j)
            curr_mag_raw = last_mag_raw + diff_raw
            
            # 更新特征向量 (21维适配)
            real_feat[:, :, 0:10] = (curr_mag_raw - m_min) / (m_max - m_min + 1e-7)
            real_feat[:, :, 10:20] = (1 - evolution_ratio) * real_feat[:, :, 10:20] + evolution_ratio * ((diff_raw - d_min) / (d_max - d_min + 1e-7))
            current_x = real_feat
            last_mag_raw = curr_mag_raw
            
        out_step, hx, buffer = model.forward_step(current_x, hx, buffer)
        # 记录时保留 NaN 可能性，用于论证崩溃
        all_outputs.append(out_step) 
        last_pred_out = out_step 
        
    return torch.cat(all_outputs, dim=1)

# ==========================================
# 3. 数据加载 (适配 21 维输入)
# ==========================================
def load_and_preprocess_degenerate(file_path, zero_pad_len=100):
    df = pd.read_csv(file_path).round(8).astype(np.float32)
    times = df['Time'].values
    dts = torch.tensor(np.diff(times, prepend=times[0]), dtype=torch.float32).unsqueeze(1)
    mag_cols = [col for col in df.columns if 'Mag' in col]
    raw_mags = df[mag_cols].values
    
    m_min, m_max = raw_mags.min(axis=0), raw_mags.max(axis=0)
    norm_mags = (raw_mags - m_min) / (m_max - m_min + 1e-7)
    
    raw_diff = np.zeros_like(raw_mags)
    raw_diff[1:] = raw_mags[1:] - raw_mags[:-1]
    d_min, d_max = raw_diff.min(axis=0), raw_diff.max(axis=0)
    norm_diff = (raw_diff - d_min) / (d_max - d_min + 1e-7)
    
    X_raw = torch.cat([torch.tensor(norm_mags), torch.tensor(norm_diff), dts], dim=1)
    Y_raw = torch.tensor(norm_mags)
    
    zero_X = torch.zeros((zero_pad_len, 21))
    zero_X[:, -1] = dts.mean()
    zero_Y = torch.zeros((zero_pad_len, 10))
    
    X = torch.cat([zero_X, X_raw], dim=0).unsqueeze(0)
    Y = torch.cat([zero_Y, Y_raw], dim=0).unsqueeze(0)
    
    norm_params = {'m_min': m_min, 'm_max': m_max, 'd_min': d_min, 'd_max': d_max}
    return X, Y, zero_pad_len, norm_params
def plot_ablation_residual_single_axis(real_y, s_out, EVOLUTION_RATIO, y_max=0.6):
    """
    在单一绝对坐标系内重叠绘制多频段残差流形 (Overlapped Manifold)
    """
    # 1. 数据解包与展平
    if isinstance(real_y, torch.Tensor):
        real_y = real_y.detach().cpu().numpy()
    if isinstance(s_out, torch.Tensor):
        s_out = s_out.detach().cpu().numpy()
        
    if s_out.ndim == 3: s_out = s_out[0]
    if real_y.ndim == 3: real_y = real_y[0]

    # 计算原始残差矩阵
    residuals = s_out - real_y
    
    # 2. 计算 RMSE 指标
    channel_rmse = np.sqrt(np.mean(residuals**2, axis=0))
    total_rmse = np.sqrt(np.mean(residuals**2))

    # 3. 开始绘图 (同轴重叠风格)
    plt.figure(figsize=(12, 8))
    
    # 使用 turbo 或 jet 这种宽光谱调色盘，确保在同一张图里 10 条线清晰可辨
    colors = plt.cm.turbo(np.linspace(0, 0.95, 10))

    # 循环绘制 10 条残差演化线
    for i in range(10): 
        plt.plot(residuals[:, i], 
                 color=colors[i], 
                 linewidth=1.2, 
                 alpha=0.85, # 略微透明以展现交织感
                 label=f'H{i+1} (RMSE: {channel_rmse[i]:.4f})')

    # 绘制理想零线
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Ideal Healthy Zero')
    # 绘制安全噪声容限带
    plt.axhspan(-0.005, 0.005, color='gray', alpha=0.2, label='Noise Floor Tolerance')

    # 核心限制：绝对固定 Y 轴边界
    plt.ylim(-y_max, y_max)
    plt.ylabel('Absolute Energy Deviation ($\delta M$)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=14, fontweight='bold')
    
    # 标题设计
    plt.title(f"Ablation Model Overlapped Residual Manifold (Ratio={EVOLUTION_RATIO})\n"
              f"(Total RMSE: {total_rmse:.5f} | Fixed Bounds \u00B1{y_max})", 
              fontsize=16, fontweight='bold', pad=15)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 图例设计：放置在图表右侧外部 (bbox_to_anchor)，绝不遮挡流形线条
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Harmonics Error", framealpha=0.9)
    plt.tight_layout()

    # 4. 自动保存
    os.makedirs('ablation_results', exist_ok=True)
    save_name = f'Ablation_Overlap_Ratio_{int(EVOLUTION_RATIO*100)}.png'
    save_path = os.path.join('ablation_results', save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print("-" * 50)
    print(f"📊 演化比例 (Ratio): {EVOLUTION_RATIO}")
    print(f"📉 全局均方根误差 (Total RMSE): {total_rmse:.6f}")
    print(f"📸 同轴重叠残差图像已保存至: {save_path}")
    print("-" * 50)
    
    plt.show()
    
    return total_rmse
# ==========================================
# 4. 执行验证
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 这里填入你残疾模型的权重路径
    MODEL_PATH = r"D:\ai\models\checkpointcanji\Evolved_Checkpoint_Ep4081.pth"
    TARGET_PIPE = r"D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv" 
    
    # 【核心变量】你可以随意调节这个 Ratio 来观察崩溃
    EVOLUTION_RATIO =1
    
    print(f"📦 正在加载【残疾模型】: {MODEL_PATH} | Ratio: {EVOLUTION_RATIO}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    model = AblationNoSigmaLNN(hidden_units=224).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X, Y, pad_len, _ = load_and_preprocess_degenerate(TARGET_PIPE)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    norm_params = checkpoint['norm_params']
    
    print(f"🚀 启动推演 (Ratio = {EVOLUTION_RATIO}) ...")
    with torch.no_grad():
        student_y = generate_self_recursive_degenerate(model, X, EVOLUTION_RATIO, norm_params, warmup_steps=pad_len)

    s_out = student_y[0, pad_len:, :].cpu().numpy()
    real_y = Y[0, pad_len:, :].cpu().numpy()
    plot_ablation_residual_single_axis(real_y, s_out, EVOLUTION_RATIO, y_max=0.6)
   