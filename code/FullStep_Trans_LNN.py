import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from ncps.torch import CfC
from ncps.wirings import FullyConnected
import math
import copy
torch.set_num_threads(8)

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
    X_raw_data = X_raw_data[:seq_len, :]
    Y_raw_data = Y_raw_data[:seq_len, :]
    
    zero_X = torch.zeros((zero_pad_len, 41), dtype=torch.float32)
    zero_X[:, -1] = dts_tensor.mean() 
    zero_Y = torch.zeros((zero_pad_len, 10), dtype=torch.float32)
    
    X = torch.cat([zero_X, X_raw_data], dim=0).unsqueeze(0)
    Y = torch.cat([zero_Y, Y_raw_data], dim=0).unsqueeze(0)
    
    norm_params = {'f_min': f_min, 'f_max': f_max, 'd_min': d_min, 'd_max': d_max, 'm_min': m_min, 'm_max': m_max}
    return X, Y, zero_pad_len, times[:seq_len], norm_params

# ==========================================
# 1. 模型定义：增加单步推理接口 forward_step
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
        batch_size, seq_len, _ = x.shape
        mags_all = x[:, :, :20] 
        diffs = x[:, :, 20:40]
        dt = x[:, :, 40:]
        
        mags_to_pool = mags_all[:, :, :10].transpose(1, 2)
        mags_padded = F.pad(mags_to_pool, (7, 7), mode='replicate') 
        smooth_bias = F.avg_pool1d(mags_padded, kernel_size=15, stride=1).transpose(1, 2)
        
        pairs = torch.stack([mags_all, diffs], dim=-1)
        channel_feats = torch.relu(self.channel_projector(pairs))
        channel_feats = self.layer_norm(channel_feats)
        attn_in = channel_feats.view(-1, self.num_channels, self.channel_dim)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        combined_feat = attn_out.reshape(batch_size, seq_len, -1)
        
        ltc_in = torch.cat([combined_feat, dt], dim=-1)
        out, _ = self.ltc(ltc_in)
        return self.fc(out) + smooth_bias

# ==========================================
# 2. 自演化递归引擎
# ==========================================
def generate_self_recursive(model, initial_X, evolution_ratio, norm_params, warmup_steps=50):
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
            current_x = initial_X[:, t:t+1, :]
            last_mag_raw = current_x[:, :, 0:10] * (m_max - m_min + 1e-7) + m_min
            last_pred_out = current_x[:, :, 0:10]
        else:
            real_feat = initial_X[:, t:t+1, :].clone()
            pred_mag_norm = last_pred_out 
            
            if evolution_ratio > 0.5:
                if evolution_ratio > 0.85:
                    noise_level = 0.0
                elif evolution_ratio > 0.8:
                    noise_level = 0.0001 * (evolution_ratio - 0.5) * 0.5
                else:
                    noise_level = 0.0001 * (evolution_ratio - 0.5)
                noise = torch.randn_like(pred_mag_norm) * noise_level
                pred_mag_norm = pred_mag_norm + noise
                
            mixed_mag_norm = (1 - evolution_ratio) * real_feat[:, :, 0:10] + evolution_ratio * pred_mag_norm
            mixed_mag_norm = torch.clamp(mixed_mag_norm, min=-0.1, max=1.2) 
            curr_mag_raw = mixed_mag_norm * (m_max - m_min + 1e-7) + m_min
            diff_raw = curr_mag_raw - last_mag_raw
            if evolution_ratio > 0.8:
                dynamic_jump = 0.01
            else:
                dynamic_jump = 0.05
                
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
        
        if torch.isnan(out_step).any():
            out_step = torch.nan_to_num(out_step, nan=0.0)  
            
        all_outputs.append(out_step)
        last_pred_out = out_step 

    return torch.cat(all_outputs, dim=1)

# ==========================================
# 3. 特化训练器
# ==========================================
class SpecializationTrainer:
    def attack_manifold_loss(self, pred, target, epoch, weight_mask=1.0):
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
     coupling_weight = 0.2
     total_loss += coupling_weight * (struct_loss * 5.0 + cos_sim_loss * 2.0)
     return total_loss

    def __init__(self, base_model_path, target_csv_path, device, resume_checkpoint_path=None):
        self.device = device
        base_checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
        self.norm_params = base_checkpoint['norm_params']
        self.teacher = AttackStageLNN().to(device)
        self.teacher.load_state_dict(base_checkpoint['model_state_dict'])
        self.teacher.eval()
        self.student = AttackStageLNN().to(device)
        self.start_epoch = 0
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            print(f" 正在从存档回滚学生模型: {resume_checkpoint_path}")
            resume_checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
            self.student.load_state_dict(resume_checkpoint['model_state_dict'])
            self.start_epoch = resume_checkpoint['epoch'] + 1 
            print(f" 时间线已恢复，将从 Epoch {self.start_epoch} 继续演化。")
        else:
            self.student.load_state_dict(base_checkpoint['model_state_dict'])
            
        from __main__ import load_and_preprocess 
        self.X, self.Y, self.pad_len, _, _ = load_and_preprocess(target_csv_path)
        self.X, self.Y = self.X.to(device), self.Y.to(device)
    def train(self, epochs=8000):
        # ==========================================
        # 1. 建立全新物理特化隔离区，防止覆盖之前的存档
        # ==========================================
        checkpoint_dir = r"D:\ai\models\checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        optimizer = optim.AdamW(self.student.parameters(), lr=0.001)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)
        
        last_loss = 1e9
        consecutive_vetoes = 0
        best_safe_weights = copy.deepcopy(self.student.state_dict())
        
        global_best_ratio = 0.0
        global_best_loss = float('inf')
        global_best_weights = copy.deepcopy(self.student.state_dict())
        
        for epoch in range(self.start_epoch, epochs):
            self.student.train()
            optimizer.zero_grad()
            
            # --- Ratio 动态爬坡曲线 ---
            if epoch <= 1800:
                progress = min(1.0, max(0.0, epoch / 1800.0))
                ratio = 0.80 * (0.5 * (1 - math.cos(progress * math.pi)))
            elif epoch <= 1900:
                ratio = 0.80
            elif epoch <= 2700:
                progress = min(1.0, (epoch - 1900) / 800.0) 
                ratio = 0.80 + 0.08 * (0.5 * (1 - math.cos(progress * math.pi)))  
            elif epoch <= 2900:
                ratio = 0.88
            else:
                progress = min(1.0, (epoch - 2900) / 2000.0) 
                ratio = 0.88 + 0.12 * (0.5 * (1 - math.cos(progress * math.pi))) 
                
            with torch.no_grad():   
                teacher_y = self.teacher(self.X)
                
            student_y = generate_self_recursive(self.student, self.X, ratio, self.norm_params)
            loss = self.attack_manifold_loss(student_y, teacher_y, epoch)
            loss.backward()
            
            # --- 终极防线：双轨级联熔断系统 ---
            if epoch > 100: 
                if loss.item() > last_loss * 4.0 or math.isnan(loss.item()) or loss.item() > 1:
                    print(f" 致命发散拦截Loss突变 ({last_loss:.6f} -> {loss.item():.6f})")
                    
                    optimizer.zero_grad() 
                    optimizer.state.clear()
                    
                    consecutive_vetoes += 1
                    
                    if consecutive_vetoes >= 3:
                        self.student.load_state_dict(global_best_weights)
                        print(f" 局部彻底死锁 (连续 {consecutive_vetoes} 次拦截)，触发【二阶硬重置】：强行注入全局最优脑！")
                        consecutive_vetoes = 0 
                        last_loss = global_best_loss
                    else:
                        self.student.load_state_dict(best_safe_weights)
                        print(f"触发【一阶局部回滚】：退回当前难度最后一次健康状态。")
                        last_loss = global_best_loss
                        
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.5, 1e-5) 
                        
                    continue 
                else:
                    consecutive_vetoes = 0 
                    if loss.item() < global_best_loss * 1.1 or loss.item() < 0.05:
                        best_safe_weights = copy.deepcopy(self.student.state_dict()) 
                    is_best = False
                    round_ratio = round(ratio, 3)
                    round_best_ratio = round(global_best_ratio, 3)
                    if round_ratio > round_best_ratio and loss.item() < 0.05:
                        global_best_ratio = round_ratio
                        global_best_loss = loss.item()
                        is_best = True
                    elif round_ratio == round_best_ratio and loss.item() < global_best_loss:
                        global_best_loss = loss.item()
                        is_best = True

                    if is_best:
                        best_path = os.path.join(checkpoint_dir, "Best_Evolved_Model.pth")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.student.state_dict(),
                            'norm_params': self.norm_params,
                            'loss': loss.item(),
                            'ratio': ratio
                        }, best_path)
                        print(f"🌟 [最优突破] 成功保存当前最强模型! Ratio: {ratio:.1%} | Loss: {loss.item():.6f}")
                        global_best_weights = copy.deepcopy(self.student.state_dict())
                    current_clip = 0.05 if loss.item() < 0.001 else 0.01
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), current_clip)
                    optimizer.step()
                    last_loss = loss.item() 
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 0.05)
                optimizer.step()
                last_loss = loss.item()

            # --- 定期切片与日志 ---
            if (epoch + 1) % 20 == 0:
                print(f" Epoch {epoch+1} 触发定期备份与波形快照...")
                backup_path = os.path.join(checkpoint_dir, f"Evolved_Checkpoint_Ep{epoch+1}.pth")            
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'norm_params': self.norm_params,                  
                    'loss': loss.item()
                }, backup_path)              
                self.plot_snapshot(student_y, teacher_y, epoch + 1)
            if epoch % 1 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Ratio: {ratio:.1%}")
        torch.save({'model_state_dict': self.student.state_dict(), 'norm_params': self.norm_params}, "Evolved_Spec.pth")
    def plot_snapshot(self, student_y, teacher_y, epoch):
        s_np = student_y[0].detach().cpu().numpy() 
        t_np = teacher_y[0].detach().cpu().numpy() 
        fig, axes = plt.subplots(10, 1, figsize=(15, 25), sharex=True)
        fig.suptitle(f'Digital Twin Evolution - Epoch {epoch}\n(Blue: Teacher Reference | Red: Student Self-Evol)', fontsize=18)    
        for i in range(10):
            axes[i].plot(t_np[:, i], color='#1f77b4', alpha=0.5, label='Teacher', linewidth=1.5)
            axes[i].plot(s_np[:, i], color='#d62728', alpha=0.8, label='Student (Evolved)', linewidth=1.2)
            axes[i].set_ylabel(f'Harmonic {i+1}', fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.5)
            if i == 0:
                axes[i].legend(loc='upper right')     
        plt.xlabel('Time Steps (4000)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        os.makedirs('training_snapshots', exist_ok=True)
        save_path = os.path.join('training_snapshots', f'Snapshot_Epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=100) 
        plt.cla()   
        plt.clf()   
        plt.close(fig) 
        print(f"📸 [Snapshot] 已保存至: {save_path}")
    def plot_comparison(self):
        self.student.eval()
        with torch.no_grad():
            t_out = self.teacher(self.X)[0, self.pad_len:, :].cpu().numpy()
            s_out = self.student(self.X)[0, self.pad_len:, :].cpu().numpy()
            real_y = self.Y[0, self.pad_len:, :].cpu().numpy()
        plt.figure(figsize=(15, 8))
        for i in range(10): 
            plt.subplot(10, 1, i + 1)
            plt.plot(real_y[:, i], 'k--', label='Ground Truth', alpha=0.4)
            plt.plot(t_out[:, i], 'b-', label='Teacher (Base)', alpha=0.6)
            plt.plot(s_out[:, i], 'r-', label='Student (Specialized)')
            plt.legend()
            plt.title(f"Harmonic {i+1} Evolution Detail")
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_MODEL = r"D:\ai\models\AttackStage_Final_Fixed.pth"
    TARGET_PIPE = r"D:\ai\BureaChurch_ADSR2_CSV\Flojtlein2\036-C_Attack.csv" 
    RESUME_ARCHIVE = r"D:\ai\models\checkpoint\Evolved_Checkpoint_Ep4900.pth" 
    trainer = SpecializationTrainer(BASE_MODEL, TARGET_PIPE, DEVICE, resume_checkpoint_path=RESUME_ARCHIVE)
    #trainer = SpecializationTrainer(BASE_MODEL, TARGET_PIPE, DEVICE) 
    trainer.train(epochs=8000)
    trainer.plot_comparison()