import numpy as np
import multiprocessing as mp
import time
import csv

# 1. 定義目標函數及其解析導數
def f(x):
    return np.cos(x)

def df_exact(x):
    return -np.sin(x)

# 2. 平行運算與單核心共用的微分核心計算
def compute_chunk(args):
    start_idx, end_idx, x_full, h = args
    chunk_size = end_idx - start_idx
    df_chunk = np.zeros(chunk_size)
    
    for i in range(chunk_size):
        idx = start_idx + i
        if 3 <= idx < len(x_full) - 3:
            df_chunk[i] = ( f(x_full[idx+3]) - 9*f(x_full[idx+2]) + 45*f(x_full[idx+1]) 
                            - 45*f(x_full[idx-1]) + 9*f(x_full[idx-2]) - f(x_full[idx-3]) ) / (60 * h)
        else:
            if 0 < idx < len(x_full) - 1:
                df_chunk[i] = (f(x_full[idx+1]) - f(x_full[idx-1])) / (2 * h)
            else:
                df_chunk[i] = 0
    return start_idx, end_idx, df_chunk

if __name__ == "__main__":
    # 初始化密集網格（一千萬個網格點）
    N = 10_000_000  
    x_full = np.linspace(0, np.pi, N)
    h = float(x_full[1] - x_full[0])
    
    # 偵測 CPU 核心數
    num_cores = mp.cpu_count()
    print("==================================================")
    print(f" 偵測到系統 CPU 核心數: {num_cores}")
    print("==================================================")
    
    # --------------------------------------------------
    # 【第一部分：16核心 平行運算】
    # --------------------------------------------------
    print(" 1. 正在啟動 [多核心平行運算]...")
    chunk_size = N // num_cores
    tasks = []
    for c in range(num_cores):
        start_idx = c * chunk_size
        end_idx = N if c == num_cores - 1 else (c + 1) * chunk_size
        tasks.append((start_idx, end_idx, x_full, h))
        
    start_time = time.time()
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(compute_chunk, tasks)
    parallel_time = time.time() - start_time
    
    # 合併平行運算數據
    df_numeric_parallel = np.zeros(N)
    for start_idx, end_idx, chunk_data in results:
        df_numeric_parallel[start_idx:end_idx] = chunk_data
        
    print(f"    -> 平行運算耗時: {parallel_time:.4f} 秒")
    
    # --------------------------------------------------
    # 【第二部分：單核心 序列運算】
    # --------------------------------------------------
    print(" 2. 正在啟動 [傳統單核心運算] 作為效能對比...")
    start_time = time.time()
    # 單核心直接計算整段網格，不切片
    _, _, df_numeric_serial = compute_chunk((0, N, x_full, h))
    serial_time = time.time() - start_time
    print(f"    -> 單核心運算耗時: {serial_time:.4f} 秒")
    
    # --------------------------------------------------
    # 【第三部分：平行運算效能與誤差報告】
    # --------------------------------------------------
    speedup = serial_time / parallel_time
    efficiency = (speedup / num_cores) * 100
    max_error = np.max(np.abs(df_numeric_parallel[3:-3] - df_exact(x_full[3:-3])))
    
    print("==================================================")
    print(" 【數值與平行效能最終報告】")
    print("==================================================")
    print(f"  網格點總數   : {N:,} 點")
    print(f"  核心加速比   : {speedup:.2f} 倍 (單核耗時 / 平行耗時)")
    print(f"  平行效率     : {efficiency:.1f} %")
    print(f"  最大絕對誤差 : {max_error:.2e}")
    print("==================================================")
    
    # --------------------------------------------------
    # 【第四部分：自動匯出 Excel (CSV) 數據表】
    # --------------------------------------------------
    print(" 3. 正在將數值微分結果匯出至 CSV 檔案...")
    csv_filename = "num_diff_results.csv"
    
    # 為了避免檔案過大（一千萬行會讓 Excel 卡死），我們每隔 5000 點抽樣一行存檔
    sample_rate = 5000
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        # 寫入 Excel 表頭
        writer.writerow(["x 座標", "解析解 (-sin x)", "16核平行趨近值", "絕對誤差"])
        
        for i in range(0, N, sample_rate):
            error_i = abs(df_numeric_parallel[i] - df_exact(x_full[i]))
            writer.writerow([
                f"{x_full[i]:.6f}", 
                f"{df_exact(x_full[i]):.6f}", 
                f"{df_numeric_parallel[i]:.6f}", 
                f"{error_i:.2e}"
            ])
            
    print(f"    -> 成功！數據表已儲存至: {csv_filename}")
    print("==================================================")
