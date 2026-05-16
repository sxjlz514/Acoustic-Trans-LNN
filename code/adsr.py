import os
import shutil
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import argrelextrema
# ================= 配置区域 =================
# 原文件夹路径
SRC_DIR = r"D:\ai\BureaChurchtest" 
# ===========================================

def process_audio_to_adsr(input_path, output_subfolder):
    """分析音频并保存 Full, Attack, Decay, Sustain, Release 五个文件"""
    try:
        # 1. 加载音频
        y, sr = librosa.load(input_path, sr=None)
        if len(y) < 1024: return False

        # 2. 提取并平滑包络 (使用 Savitzky-Golay 替代简单卷积，更平滑)
        hop_length = 256
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms = signal.savgol_filter(rms, window_length=21, polyorder=2)
        
        # 归一化
        rms_max = np.max(rms)
        rms_min = np.min(rms)
        if rms_max > rms_min:
            rms = (rms - rms_min) / (rms_max - rms_min)
        
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # 3. 逻辑切分点 (按照你的新准则)
        
       # Attack End: 直接定位到峰值
        idx_attack_end = np.argmax(rms)
        
        # Decay End: 峰值后的第一个波谷 (Local Minimum)
        # order=30 是波谷平滑参数
        valleys = argrelextrema(rms, np.less, order=30)[0]
        post_peak_valleys = valleys[valleys > idx_attack_end]
        idx_d_end = post_peak_valleys[0] if len(post_peak_valleys) > 0 else idx_attack_end + int(0.1 * len(rms))
        
        # Release Start: 最后一个 >= 80% 峰值强度的【波峰】位置
        # 1. 寻找所有局部波峰
        all_peaks = argrelextrema(rms, np.greater, order=30)[0]
        
        # 2. 筛选：必须在 Decay 之后，且强度 >= 0.8
        # rms 已经归一化，峰值为 1.0，所以 0.8 即为 80% 强度
        qualified_peaks = all_peaks[(all_peaks > idx_d_end) & (rms[all_peaks] >= 0.8)]
        
        # 3. 取最后一个符合条件的波峰
        idx_r_start = qualified_peaks[-1] if len(qualified_peaks) > 0 else int(0.9 * len(rms))
        
        # 边界安全性检查：强制保证 Attack < Decay < Sustain < Release
        if idx_d_end <= idx_attack_end: 
            idx_d_end = idx_attack_end + 1
        if idx_r_start <= idx_d_end: 
            idx_r_start = idx_d_end + 1
        if idx_r_start >= len(rms): 
            idx_r_start = len(rms) - 1

        # 转换为采样点
        def to_s(t): return int(t * sr)
        
        segments = {
            "Attack": (0, to_s(times[idx_attack_end])),
            "Decay": (to_s(times[idx_attack_end]), to_s(times[idx_d_end])),
            "Sustain": (to_s(times[idx_d_end]), to_s(times[idx_r_start])),
            "Release": (to_s(times[idx_r_start]), len(y))
        }

        base_name = os.path.splitext(os.path.basename(input_path))[0]

        # --- 执行保存 ---
        # A. 保存原始全长音频 (加上 _Full 后缀)
        full_out_path = os.path.join(output_subfolder, f"{base_name}_Full.wav")
        sf.write(full_out_path, y, sr)

        # B. 保存四个 ADSR 段 (加入边缘缓冲逻辑)
        margin_sec = 0.05  # 设置 50ms 缓冲
        margin_samples = int(margin_sec * sr)
        
        for stage, (start, end) in segments.items():
            if end > start:
                # 计算带缓冲的边界
                new_start = start - margin_samples
                new_end = end + margin_samples
                
                # 计算填充空白的长度
                pad_left = max(0, -new_start)
                pad_right = max(0, new_end - len(y))
                
                # 获取有效数据切片
                valid_slice = y[max(0, new_start) : min(len(y), new_end)]
                
                # 拼接：左填充 + 有效数据 + 右填充
                if pad_left > 0 or pad_right > 0:
                    seg_data = np.concatenate([
                        np.zeros(pad_left, dtype=y.dtype),
                        valid_slice,
                        np.zeros(pad_right, dtype=y.dtype)
                    ])
                else:
                    seg_data = valid_slice
                
                out_name = f"{base_name}_{stage}.wav"
                sf.write(os.path.join(output_subfolder, out_name), seg_data, sr)
        
        return True
    except Exception as e:
        print(f"处理失败 {input_path}: {e}")
        return False

def run_mirror_task():
    if not os.path.exists(SRC_DIR):
        print(f"错误：找不到原文件夹 {SRC_DIR}")
        return

    src_abs = os.path.abspath(SRC_DIR)
    # 目标目录添加 _ADSR 后缀
    dst_root = src_abs + "_ADSR2"
    
    audio_exts = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')

    print(f">>> 任务启动")
    print(f">>> 源: {src_abs} \n>>> 目标: {dst_root}\n")

    for root, dirs, files in os.walk(src_abs):
        rel_path = os.path.relpath(root, src_abs)
        target_dir = os.path.normpath(os.path.join(dst_root, rel_path))
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for file in files:
            src_file_path = os.path.join(root, file)
            
            if file.lower().endswith(audio_exts):
                print(f"[处理+保存全长] {file}")
                process_audio_to_adsr(src_file_path, target_dir)
            else:
                # 非音频文件（如说明文档、图片）直接拷贝
                dst_file_path = os.path.join(target_dir, file)
                shutil.copy2(src_file_path, dst_file_path)
                print(f"[拷贝附件] {file}")

    print(f"\n>>> 任务全部完成！")

if __name__ == "__main__":
    run_mirror_task()