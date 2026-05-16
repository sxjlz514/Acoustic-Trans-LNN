import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math
import copy
import csv
from ncps.torch import CfC
from ncps.wirings import FullyConnected

# 设置计算环境
torch.set_num_threads(8)

# ==========================================
# 1. 消融架构模型 (21维输入: 10 Mag + 10 Diff + 1 dt)
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
        """单步推理接口：用于自演化演推 [cite: 2026-05-05]"""
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

    def forward(self, x):
        """并行训练接口"""
        batch_size, seq_len, _ = x.shape
        mags = x[:, :, :10]; diffs = x[:, :, 10:20]; dt = x[:, :, 20:]
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
        return self.fc(out) + smooth_bias

# ==========================================
# 2. 消融版专属工具函数
# ==========================================
def load_and_preprocess_degenerate(file_path, zero_pad_len=100):
    df = pd.read_csv(file_path).round(8).astype(np.float32)
    times = df['Time'].values
    dts = torch.tensor(np.diff(times, prepend=times[0]), dtype=torch.float32).unsqueeze(1)
    mag_cols = [col for col in df.columns if 'Mag' in col]
    raw_mags = df[mag_cols].values
    
    # 归一化
    m_min, m_max = raw_mags.min(axis=0), raw_mags.max(axis=0)
    norm_mags = (raw_mags - m_min) / (m_max - m_min + 1e-7)
    
    # 差分
    raw_diff = np.zeros_like(raw_mags)
    raw_diff[1:] = raw_mags[1:] - raw_mags[:-1]
    d_min, d_max = raw_diff.min(axis=0), raw_diff.max(axis=0)
    norm_diff = (raw_diff - d_min) / (d_max - d_min + 1e-7)
    
    X_raw = torch.cat([torch.tensor(norm_mags), torch.tensor(norm_diff), dts], dim=1)
    Y_raw = torch.tensor(norm_mags)
    
    zero_X = torch.zeros((zero_pad_len, 21)); zero_X[:, -1] = dts.mean()
    zero_Y = torch.zeros((zero_pad_len, 10))
    
    X = torch.cat([zero_X, X_raw], dim=0).unsqueeze(0)
    Y = torch.cat([zero_Y, Y_raw], dim=0).unsqueeze(0)
    
    norm_params = {'m_min': m_min, 'm_max': m_max, 'd_min': d_min, 'd_max': d_max}
    return X, Y, zero_pad_len, norm_params

def generate_self_recursive_degenerate(model, initial_X, evolution_ratio, norm_params, warmup_steps=50):
    """消融版递归引擎：缺失 Sigma 项引导 [cite: 2026-05-05]"""
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
       current_x = initial_X[:, t:t+1, :].clone() # 必须 clone 以断开版本计数
       last_mag_raw = current_x[:, :, 0:10] * (m_max - m_min + 1e-7) + m_min
       last_pred_out = current_x[:, :, 0:10]
     else:
       # 彻底重新构建张量，严禁对 real_feat 进行索引赋值
       real_feat = initial_X[:, t:t+1, :].clone()
       pred_mag_norm = last_pred_out 
    
       # 混合逻辑
       mixed_mag_norm = (1 - evolution_ratio) * real_feat[:, :, 0:10] + evolution_ratio * pred_mag_norm
       curr_mag_raw = torch.clamp(mixed_mag_norm, -0.1, 1.2) * (m_max - m_min + 1e-7) + m_min
       diff_raw = curr_mag_raw - last_mag_raw
    
       max_j = (m_max - m_min) * (0.01 if evolution_ratio > 0.8 else 0.05)
       diff_raw = torch.clamp(diff_raw, -max_j, max_j)
       curr_mag_raw = last_mag_raw + diff_raw

       norm_mag = (curr_mag_raw - m_min) / (m_max - m_min + 1e-7)
       norm_diff = (1 - evolution_ratio) * real_feat[:, :, 10:20] + evolution_ratio * ((diff_raw - d_min) / (d_max - d_min + 1e-7))
       dt_val = real_feat[:, :, 20:]
    
       current_x = torch.cat([norm_mag, norm_diff, dt_val], dim=-1)
       last_mag_raw = curr_mag_raw
            
       out_step, hx, buffer = model.forward_step(current_x, hx, buffer)
       all_outputs.append(torch.nan_to_num(out_step, nan=0.0))
       last_pred_out = out_step 
    return torch.cat(all_outputs, dim=1)

# ==========================================
# 3. 消融组特化训练器 (严格隔离路径)
# ==========================================
class SpecializationTrainer:
    def __init__(self, teacher_model_path, student_model_path, target_csv_path, device, resume_checkpoint_path=None):
        self.device = device
        # Teacher 和 Student 均使用 AblationNoSigmaLNN 架构 [cite: 2026-05-05]
        base_ckpt = torch.load(teacher_model_path, map_location=device, weights_only=False)
        self.norm_params = base_ckpt['norm_params']
        
        self.teacher = AblationNoSigmaLNN(input_dim=21, hidden_units=224).to(device)
        self.teacher.load_state_dict(base_ckpt['model_state_dict'])
        self.teacher.eval()
        
        self.student = AblationNoSigmaLNN(input_dim=21, hidden_units=224).to(device)
        self.start_epoch = 0
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
            self.student.load_state_dict(ckpt['model_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
        else:
            self.student.load_state_dict(base_ckpt['model_state_dict'])
            
        self.X, self.Y, self.pad_len, self.norm_params = load_and_preprocess_degenerate(target_csv_path)
        self.X, self.Y = self.X.to(device), self.Y.to(device)

    def attack_manifold_loss(self,pred, target, epoch, weight_mask=1):
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
          
          

    def train(self, epochs=4900):
        checkpoint_dir = r"D:\ai\models\checkpointcanji"
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_csv = os.path.join(checkpoint_dir, "evolution_history_log.csv")
        
        if self.start_epoch == 0:
            with open(log_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['Epoch', 'Ratio', 'Loss'])

        optimizer = optim.AdamW(self.student.parameters(), lr=0.001)
        last_loss = 1e9
        global_best_ratio, global_best_loss = 0.0, float('inf')
        global_best_weights = copy.deepcopy(self.student.state_dict())
        best_safe_weights = copy.deepcopy(self.student.state_dict())
        consecutive_vetoes = 0

        for epoch in range(self.start_epoch, epochs):
            self.student.train()
            optimizer.zero_grad()
            
            # --- 分段 Ratio 爬坡 ---
            if epoch <= 1800: ratio = 0.80 * (0.5 * (1 - math.cos(epoch / 1800.0 * math.pi)))
            elif epoch <= 1900: ratio = 0.80
            elif epoch <= 2700: ratio = 0.80 + 0.08 * (0.5 * (1 - math.cos((epoch-1900)/800 * math.pi)))
            elif epoch <= 2900: ratio = 0.88
            else: ratio = 0.88 + 0.12 * (0.5 * (1 - math.cos((epoch-2900)/2000 * math.pi)))
                
            with torch.no_grad(): teacher_y = self.teacher(self.X)
            student_y = generate_self_recursive_degenerate(self.student, self.X, ratio, self.norm_params)
            min_len = min(student_y.size(1), teacher_y.size(1))
            student_y_clipped = student_y[:, :min_len, :]
            teacher_y_clipped = teacher_y[:, :min_len, :]
            loss = self.attack_manifold_loss(student_y_clipped, teacher_y_clipped, epoch)
            
            # --- 双轨熔断 ---
            if epoch > 100 and (loss.item() > last_loss * 4.0 or math.isnan(loss.item()) or loss.item() > 1.5):
                print(f"🛑 崩溃拦截 @ Epoch {epoch} Ratio {ratio:.1%}")
                optimizer.state.clear(); consecutive_vetoes += 1
                self.student.load_state_dict(global_best_weights if consecutive_vetoes >= 3 else best_safe_weights)
                for g in optimizer.param_groups: g['lr'] = max(g['lr']*0.5, 1e-5)
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 0.01)
            optimizer.step()
            
            # 记录与存档
            last_loss = loss.item()
            with open(log_csv, 'a', newline='') as f: csv.writer(f).writerow([epoch, ratio, last_loss])
            
            if ratio >= global_best_ratio and last_loss < 0.1:
                global_best_ratio, global_best_loss = ratio, last_loss
                global_best_weights = copy.deepcopy(self.student.state_dict())
                torch.save({'epoch': epoch, 'model_state_dict': self.student.state_dict(), 'norm_params': self.norm_params}, 
                           os.path.join(checkpoint_dir, "Best_Evolved_Model.pth"))

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1} | Loss: {last_loss:.6f} | Ratio: {ratio:.1%}")
            if epoch  % 20 == 0:
                torch.save({'epoch': epoch, 'model_state_dict': self.student.state_dict(), 'norm_params': self.norm_params}, 
                           os.path.join(checkpoint_dir, f"Evolved_Checkpoint_Ep{epoch+1}.pth"))

    def plot_comparison(self):
        self.student.eval()
        with torch.no_grad():
            t_out = self.teacher(self.X)[0, self.pad_len:, :].cpu().numpy()
            s_out = generate_self_recursive_degenerate(self.student, self.X, 1.0, self.norm_params)[0, self.pad_len:, :].cpu().numpy()
        plt.figure(figsize=(12, 10))
        for i in range(10):
            plt.subplot(5, 2, i+1); plt.plot(t_out[:, i], 'b', alpha=0.5); plt.plot(s_out[:, i], 'r')
            plt.title(f"H{i+1}"); plt.grid(True)
        plt.tight_layout(); plt.show()

def plot_erl_curves(log_path):
    data = pd.read_csv(log_path)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data['Epoch'], data['Ratio'], 'b-', label='Ratio'); ax1.set_ylabel('Ratio', color='b')
    ax2 = ax1.twinx(); ax2.plot(data['Epoch'], data['Loss'], 'r-', alpha=0.5, label='Loss')
    ax2.set_ylabel('Loss', color='r'); ax2.set_yscale('log')
    plt.title("ERL Analysis (Ablation Group)"); plt.show()

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_MODEL_CANJI = r"D:\ai\models\AttackStage_Ablation_SingleStep_50.pth"
    TARGET_PIPE = r"D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv" 
    RESUME_ARCHIVE = r"D:\ai\models\checkpointcanji\Evolved_Checkpoint_Ep2901.pth" 
    LOG_CSV = r"D:\ai\models\checkpointcanji\evolution_history_log.csv"

    trainer = SpecializationTrainer(BASE_MODEL_CANJI, BASE_MODEL_CANJI, TARGET_PIPE, DEVICE, 
                                    resume_checkpoint_path=RESUME_ARCHIVE if os.path.exists(RESUME_ARCHIVE) else None)
    try:
        trainer.train(epochs=4920)
    except KeyboardInterrupt:
        print("停止训练，绘制曲线...")
    
    plot_erl_curves(LOG_CSV)
    trainer.plot_comparison()