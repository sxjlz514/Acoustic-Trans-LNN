import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import os

# ================= 配置区域 =================
SRC_DIR = r"D:\ai\BureaChurchtest_ADSR2"  # ADSR 音频根目录
NUM_HARMONICS = 10 
CYCLES_PER_WINDOW = 6 

# 定义不同阶段的目标行数 (物理适配)
ROWS_MAP = {
    'Attack': 4000,
    'Decay': 500,
    'Release': 4000
}
# ===========================================

def estimate_f0_robust(y, sr):
    n_fft = 2**int(np.ceil(np.log2(len(y))))
    spectrum = np.abs(np.fft.rfft(y, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    return max(freqs[np.argmax(spectrum)], 20.0)

def extract_to_csv_adaptive(file_path, output_path, target_rows):
    """核心提取：计算全长特征，并仅在有效物理区间内重采样至 target_rows"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. 定义缓冲逻辑
        margin_sec = 0.05 # 50ms 缓冲
        total_duration = len(y) / sr
        
        # 确定有效提取区间
        if total_duration > (margin_sec * 2):
            valid_start = margin_sec
            valid_end = total_duration - margin_sec
        else:
            # 如果音频过短，回退到原始长度，不做切除
            valid_start = 0
            valid_end = total_duration
            
        file_len = len(y)
        if file_len < 128: return False
        
        f0 = estimate_f0_robust(y, sr)
        
        # 2. 计算采样逻辑 (保持原有逻辑)
        win_len = int((sr / f0) * CYCLES_PER_WINDOW)
        win_len = max(128, min(win_len, 8192))
        exec_hop = max(1, int(np.floor((file_len - win_len) / target_rows)))
        frame_indices = np.arange(0, file_len - win_len + 1, exec_hop)
        
        if len(frame_indices) < 2: return False
        
        # 原始计算的时间戳 (包含了 Padding 部分，这是为了插值计算的准确性)
        actual_times = (frame_indices + win_len // 2) / sr
        
        # 【核心改动】：target_times 强制锁定在有效物理区间
        target_times = np.linspace(valid_start, valid_end, target_rows)
        final_data = {'Time': target_times}

        # 3. 特征提取 (保持原有的滤波与希尔伯特变换逻辑)
        for k in range(1, NUM_HARMONICS + 1):
            freq_target = k * f0
            if freq_target >= sr / 2: break
            low = max(0.1, freq_target - f0/4)
            high = min(sr/2 - 1, freq_target + f0/4)
            sos = signal.butter(4, [low, high], btype='bandpass', fs=sr, output='sos')
            filtered_y = signal.sosfiltfilt(sos, y) # 这里用的是带 Padding 的 y
            analytic_signal = signal.hilbert(filtered_y)
            amplitude_envelope = np.abs(analytic_signal)
            log_amplitude = np.log(amplitude_envelope + 1e-9)
            sigma_full = np.gradient(log_amplitude, 1/sr)
            smooth_win = int(sr / f0)
            if smooth_win % 2 == 0: smooth_win += 1
            if len(sigma_full) > smooth_win > 5:
                sigma_full = signal.savgol_filter(sigma_full, smooth_win, 2)
            raw_mag = 20 * np.log10(amplitude_envelope[frame_indices] + 1e-9)
            raw_sigma = sigma_full[frame_indices]
            
            f_mag = interp1d(actual_times, raw_mag, kind='linear', fill_value="extrapolate")
            f_sigma = interp1d(actual_times, raw_sigma, kind='linear', fill_value="extrapolate")
            
            final_data[f'H{k}_Mag'] = f_mag(target_times)
            final_data[f'H{k}_Sigma'] = f_sigma(target_times)

        pd.DataFrame(final_data).to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"提取失败 {file_path}: {e}")
        return False

def run_batch():
    src_abs = os.path.abspath(SRC_DIR)
    dst_root = src_abs + "_CSV"
    
    print(f"开始物理动力学特征提取...")
    
    for root, dirs, files in os.walk(src_abs):
        rel_path = os.path.relpath(root, src_abs)
        target_dir = os.path.join(dst_root, rel_path)
        if not os.path.exists(target_dir): os.makedirs(target_dir)

        for file in files:
            if not file.lower().endswith(('.wav', '.flac')): continue
            
            name_without_ext = os.path.splitext(file)[0]
            
            # 根据后缀匹配 target_rows
            target_rows = None
            for key, val in ROWS_MAP.items():
                if name_without_ext.endswith(key):
                    target_rows = val
                    break
            
            # 如果不匹配 (比如是 _Sustain 或 _Full)，则跳过
            if target_rows is None:
                continue

            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, name_without_ext + ".csv")
            
            print(f"正在处理 [{key}]: {file}")
            if extract_to_csv_adaptive(src_path, dst_path, target_rows):
                print(f"  [DONE] -> {target_rows} rows")

    print(f"\n任务完成CSV 矩阵已存至: {dst_root}")

if __name__ == "__main__":
    run_batch()