import numpy as np
import multiprocessing as mp
import time

# 1. 定義目標函數及其解析導數
def f(x):
    return np.cos(x)

def df_exact(x):
    return -np.sin(x)

# 2. 平行運算的子任務：計算特定網格區間的六階中心差分
def compute_chunk(args):
    start_idx, end_idx, x_full, h = args
    chunk_size = end_idx - start_idx
    df_chunk = np.zeros(chunk_size)
    
    for i in range(chunk_size):
        idx = start_idx + i
        
        # 【終極修正】六階中心差分正確係數與正負號
        if 3 <= idx < len(x_full) - 3:
            df_chunk[i] = ( f(x_full[idx+3]) - 9*f(x_full[idx+2]) + 45*f(x_full[idx+1]) 
                            - 45*f(x_full[idx-1]) + 9*f(x_full[idx-2]) - f(x_full[idx-3]) ) / (60 * h)
        # 邊界點處理（靠近 0 或 pi 處，使用標準中心差分）
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
    
    # 精確計算兩點之間的間距常數 h
    h = x_full[1] - x_full[0]  
    
    # 設定平行運算核心數與任務切片 (Chunking)
    num_cores = mp.cpu_count()
    print(f"==================================================")
    print(f" 偵測到系統 CPU 核心數: {num_cores}，開始平行運算...")
    print(f"==================================================")
    
    chunk_size = N // num_cores
    tasks = []
    for c in range(num_cores):
        start_idx = c * chunk_size
        end_idx = N if c == num_cores - 1 else (c + 1) * chunk_size
        tasks.append((start_idx, end_idx, x_full, h))
        
    # 啟動多進程平行計算
    start_time = time.time()
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(compute_chunk, tasks)
    parallel_time = time.time() - start_time
    
    # 合併各核心計算出來的數據結果
    df_numeric = np.zeros(N)
    for start_idx, end_idx, chunk_data in results:
        df_numeric[start_idx:end_idx] = chunk_data
        
    print(f" 網格點總數: {N:,} 點")
    print(f" 平行計算總耗時: {parallel_time:.4f} 秒")
    
    # 驗證內部點的最大數值絕對誤差
    max_error = np.max(np.abs(df_numeric[3:-3] - df_exact(x_full[3:-3])))
    print(f" 內部點的最大絕對誤差 (vs -sin(x)): {max_error:.2e}")
    print(f"==================================================")
