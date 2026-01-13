import numpy as np # 数值计算
import pandas as pd # 数据处理
import matplotlib.pyplot as plt # 绘图
plt.switch_backend('Agg')  # 非交互式绘图，解决主线程问题的核心配置
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, resample, correlate # 信号处理（相关性、重采样）
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
from sklearn.multioutput import MultiOutputRegressor # 多输出回归包装器
from sklearn.preprocessing import StandardScaler # 特征标准化
from sklearn.metrics import r2_score # 模型评估（R²）
import os # 路径处理
import joblib  # 模型保存/加载

# ==========================================
# 0. 配置
# ==========================================
# 绘图字体（解决中文/负号显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 1. 扩充后的波形数据目录 (包含原始CSV和工况表)
DATA_DIR = r'C:\Users\xanthe\Desktop\code\augmented_data' # 波形数据目录

# 2. 扩充后的工况表路径
COND_PATH = os.path.join(DATA_DIR, 'Augmented_Conditions.csv') # 工况表

# 3. 重新提取后的纯净特征参数表路径
PARAM_PATH = r'augmented_extraction_results\Extracted_Parameters.csv' # 波形参数表

# 4. 预测结果输出目录
OUTPUT_DIR = "augmented_prediction_results_rf"  # 预测结果输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. 核心工具：骨架生成与相位对齐
# ==========================================
# 合成波形生成 generate_synthetic_waveform
def generate_synthetic_waveform(total_time, dt, templates):
    """
    根据预测的 Duration (转化为 w) 按 P1-P2-P3 循环生成。
    """
    t_generated = np.arange(0, total_time, dt)
    y_generated = np.zeros_like(t_generated)

    current_t = 0
    idx = 0
    pattern_idx = 0
    segments = []

    # 1. 循环生成片段
    while idx < len(t_generated):
        temp = templates[pattern_idx % 3]
        seg_dur = temp['dur']
        w = 2 * np.pi / seg_dur

        start_idx = idx
        end_idx = min(len(t_generated), int(start_idx + seg_dur / dt))

        if start_idx >= end_idx: break

        def make_predictor(A, w, B, start_time):
            # 默认相位 -pi/2，保证波形从波谷平滑开始
            return lambda t_g: A * np.sin(w * (t_g - start_time) - np.pi / 2) + B

        predictor = make_predictor(temp['A'], w, temp['B'], current_t)
        segments.append({'func': predictor, 'start': start_idx, 'end': end_idx})

        t_slice = t_generated[start_idx:end_idx]
        y_generated[start_idx:end_idx] = predictor(t_slice)

        current_t += seg_dur
        idx = end_idx
        pattern_idx += 1

    # 2. 平滑混合
    blend_steps = 4
    for i in range(len(segments) - 1):
        curr = segments[i]
        next_s = segments[i + 1]
        join_idx = curr['end']
        b_start = max(0, join_idx - blend_steps)
        b_end = min(len(t_generated), join_idx + blend_steps)
        if b_end - b_start < 2: continue

        t_blend = t_generated[b_start:b_end]
        y_left = curr['func'](t_blend)
        y_right = next_s['func'](t_blend)
        weights = np.linspace(0, 1, len(t_blend))
        y_generated[b_start:b_end] = (1 - weights) * y_left + weights * y_right

    return t_generated, y_generated

# 相位对齐 align_phase_offset
def align_phase_offset(y_true, y_pred):
    """ 相位自动对齐工具 """
    """通过互相关分析找到最优滞后，对齐预测波形与真实波形的相位"""
    if len(y_true) == 0 or len(y_pred) == 0: return y_pred

    correlation = correlate(y_true - np.mean(y_true), y_pred - np.mean(y_pred), mode='full')
    lags = np.arange(-len(y_true) + 1, len(y_true))
    best_lag = lags[np.argmax(correlation)] # 最大相关性对应的滞后值

    if best_lag > 0:
        y_aligned = np.pad(y_pred, (best_lag, 0), mode='edge')[:len(y_pred)]
    else:
        y_aligned = np.pad(y_pred, (0, -best_lag), mode='edge')[-len(y_pred):]

    return y_aligned # 按最优滞后填充/截断波形，实现相位对齐

# DTW 距离计算 compute_dtw_metric
def compute_dtw_metric(s1, s2, resample_len=100):
    """计算归一化动态时间规整（DTW）距离，评估波形相似性"""
    s1_res = resample(s1, resample_len)
    s2_res = resample(s2, resample_len)
    n = len(s1_res)
    m = len(s2_res)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1_res[i - 1] - s2_res[j - 1])
            last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix[n, m] / resample_len# 先重采样到相同长度，再构建DTW矩阵，计算最小累积距离

# 数据加载 load_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 't' in df.columns and 'y' in df.columns:
            return df['t'].values, df['y'].values
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except:
        return None, None

# ==========================================
# 2. 数据准备 prepare_dataset
# ==========================================
def prepare_dataset():
    # 加载工况表+波形参数表，按FileName合并
    print("正在准备数据集 (RF Version)...")
    if not os.path.exists(COND_PATH): raise FileNotFoundError(f"找不到工况表: {COND_PATH}")
    if not os.path.exists(PARAM_PATH): raise FileNotFoundError(f"找不到参数表: {PARAM_PATH}")

    df_cond = pd.read_csv(COND_PATH)
    df_params = pd.read_csv(PARAM_PATH)
    df_merged = pd.merge(df_cond, df_params, on='FileName', how='inner')
    print(f"有效样本数: {len(df_merged)}")
    
    # 推导P1/P2/P3的最大值（B+A）、最小值（B-A）
    for k in range(1, 4):
        df_merged[f'P{k}_Max'] = df_merged[f'P{k}_B'] + df_merged[f'P{k}_A']
        df_merged[f'P{k}_Min'] = df_merged[f'P{k}_B'] - df_merged[f'P{k}_A']
    
    # 定义特征列（工况）和目标列（波形参数）
    feature_cols = ['h', 'd', 'lambda', 'rho']
    target_cols = []
    for k in range(1, 4):
        target_cols.extend([f'P{k}_Max', f'P{k}_Min', f'P{k}_dur'])

    print(f"预测目标列: {target_cols}")
    return df_merged, feature_cols, target_cols

# ==========================================
# 3. LOGO 验证 run_logo_validation
# ==========================================
def run_logo_validation(df, feature_cols, target_cols):
    print("\n" + "=" * 50)
    print("开始执行 Leave-One-Group-Out (LOGO) 验证")
    print("策略: Random Forest (骨架+血肉预测) + 自动对齐")
    print("=" * 50)

    unique_originals = df['OriginalFile'].unique() # 按原始文件分组
    summary_results = []
    all_true = []
    all_pred = []

    for i, orig_file in enumerate(unique_originals):
         # 划分测试集（当前组）和训练集（其余组）
        print(f"[{i + 1}/{len(unique_originals)}] 验证组: {orig_file} ... ", end="")

        test_mask = df['OriginalFile'] == orig_file
        df_test = df[test_mask]
        df_train = df[~test_mask]

        if len(df_test) == 0: continue

        X_train = df_train[feature_cols]
        Y_train = df_train[target_cols]
        X_test = df_test[feature_cols]
        Y_test = df_test[target_cols]
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练多输出随机森林模型
        # n_jobs=-1 表示并行使用所有CPU核心
        regr_multi = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        )
        regr_multi.fit(X_train_scaled, Y_train)

        # 预测目标参数（Max/Min/dur）
        Y_pred_group = regr_multi.predict(X_test_scaled)
        all_true.extend(Y_test.values)
        all_pred.extend(Y_pred_group)

        # 基于预测参数生成合成波形，相位对齐后评估
        group_r2 = []
        group_nmae = []
        group_dtw = []

        for idx in range(len(df_test)):
            # 1. 解析预测参数，构建P1/P2/P3模板
            sample_row = df_test.iloc[idx]
            fname = sample_row['FileName']

            pred_vals = pd.Series(Y_pred_group[idx], index=target_cols)
            predicted_templates = []
            for k in range(1, 4):
                try:
                    p_max = pred_vals[f'P{k}_Max']
                    p_min = pred_vals[f'P{k}_Min']
                    p_dur = pred_vals[f'P{k}_dur']
                    predicted_templates.append({
                        'id': k,
                        'A': (p_max - p_min) / 2,
                        'B': (p_max + p_min) / 2,
                        'dur': max(0.05, p_dur)
                    })
                except: pass

            t_true, y_true = load_data(os.path.join(DATA_DIR, fname))
            if t_true is None: continue

            # 2. 生成合成波形
            total_time = t_true[-1] - t_true[0]
            dt = t_true[1] - t_true[0] if len(t_true) > 1 else 0.01
            t_gen, y_gen = generate_synthetic_waveform(total_time + dt, dt, predicted_templates)

            min_len = min(len(y_true), len(y_gen))
            y_true_cut = y_true[:min_len]
            y_gen_cut = y_gen[:min_len]
            t_cut = t_true[:min_len]

            # 3. 相位对齐
            y_aligned = align_phase_offset(y_true_cut, y_gen_cut)
            
            # 4. 计算评估指标（R²、NMAE、DTW）
            mask = ~np.isnan(y_true_cut) & ~np.isnan(y_aligned)
            if np.sum(mask) > 10:
                y_t_val = y_true_cut[mask]
                y_p_val = y_aligned[mask]
                group_r2.append(r2_score(y_t_val, y_p_val))
                residuals = np.abs(y_t_val - y_p_val)
                rng = np.max(y_t_val) - np.min(y_t_val)
                group_nmae.append(np.max(residuals) / rng if rng > 0 else 0)
                group_dtw.append(compute_dtw_metric(y_t_val, y_p_val))

            # 5. 绘制波形对比图并保存
            if idx == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(t_cut, y_true_cut, 'k', alpha=0.3, label='Truth')
                plt.plot(t_cut, y_aligned, 'r--', linewidth=1.5, label='RF Prediction')
                dur_info = [f"{p['dur']:.3f}s" for p in predicted_templates]
                plt.title(f"{orig_file} | R2={group_r2[-1]:.4f}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"Pred_{orig_file}.png"))
                plt.close('all')  # 彻底关闭所有绘图窗口，释放资源

        avg_r2 = np.mean(group_r2) if group_r2 else 0
        avg_nmae = np.mean(group_nmae) if group_nmae else 0
        avg_dtw = np.mean(group_dtw) if group_dtw else 0
        
        # 保存每组的评估结果
        print(f"R2={avg_r2:.4f}, NMAE={avg_nmae:.2%}")
        summary_results.append({
            'OriginalFile': orig_file,
            'Avg_R2': avg_r2,
            'Avg_NMAE': avg_nmae,
            'Avg_DTW': avg_dtw
        })

    # 保存LOGO验证汇总、参数级R²评分
    df_res = pd.DataFrame(summary_results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "LOGO_Summary_Skeleton_RF.csv"), index=False)

    # ==========================================
    # 【新增】计算参数级别的 R2
    # ==========================================
    print("\n" + "=" * 50)
    print(" >>> 参数预测精度评估 (Parameter Level R2) <<<")
    print("=" * 50)

    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)

    param_r2_scores = r2_score(all_true_np, all_pred_np, multioutput='raw_values')

    df_param_perf = pd.DataFrame({
        'Parameter': target_cols,
        'R2_Score': param_r2_scores
    })

    dur_r2 = df_param_perf[df_param_perf['Parameter'].str.contains('dur')]['R2_Score'].mean()
    max_r2 = df_param_perf[df_param_perf['Parameter'].str.contains('Max')]['R2_Score'].mean()
    min_r2 = df_param_perf[df_param_perf['Parameter'].str.contains('Min')]['R2_Score'].mean()

    print(df_param_perf)
    print("-" * 30)
    print(f"周期 (Duration) 平均 R2: {dur_r2:.4f}")
    print(f"最大值 (Max) 平均 R2:    {max_r2:.4f}")
    print(f"最小值 (Min) 平均 R2:    {min_r2:.4f}")

    df_param_perf.to_csv(os.path.join(OUTPUT_DIR, "Parameter_R2_Scores_RF.csv"), index=False)
    print(f"\n结果已保存至: {OUTPUT_DIR}")
    
    # 返回最后一次训练的模型（或根据需求调整为全量训练）
    return regr_multi, scaler

# ==========================================
# 4. 全量数据训练最终模型（可选，根据需求选择）（新增）
# ==========================================
def train_final_model(df, feature_cols, target_cols):
    """使用全量数据训练最终模型并保存"""
    print("\n" + "=" * 50)
    print("开始训练全量数据的最终模型")
    print("=" * 50)
    
    X = df[feature_cols]
    Y = df[target_cols]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    final_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    final_model.fit(X_scaled, Y)
    
    # 保存模型和标准化器
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'wind_turbine_predictor.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    print(f"最终模型已保存至: {os.path.join(OUTPUT_DIR, 'wind_turbine_predictor.pkl')}")
    print(f"标准化器已保存至: {os.path.join(OUTPUT_DIR, 'scaler.pkl')}")
    
    return final_model, scaler

if __name__ == "__main__":
    try:
        # 数据准备
        df, f_cols, t_cols = prepare_dataset()
        
        # 执行LOGO验证（返回最后一次训练的模型，可选）
        logo_model, logo_scaler = run_logo_validation(df, f_cols, t_cols)
        
        # 【推荐】训练全量数据的最终模型并保存
        final_model, final_scaler = train_final_model(df, f_cols, t_cols)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()