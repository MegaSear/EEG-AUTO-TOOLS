import numpy as np
import numba


# Локальное обновление для S: обновляются только строки и столбцы p и q
@numba.njit
def local_update(S, p, q, cos_val, sin_val):
    n_matrices, n, _ = S.shape
    for k in range(n_matrices):
        # Копируем строки p и q
        row_p = np.empty(n, dtype=S.dtype)
        row_q = np.empty(n, dtype=S.dtype)
        for j in range(n):
            row_p[j] = S[k, p, j]
            row_q[j] = S[k, q, j]
        # Обновляем строки p и q
        for j in range(n):
            S[k, p, j] = cos_val * row_p[j] + sin_val * row_q[j]
            S[k, q, j] = -sin_val * row_p[j] + cos_val * row_q[j]
        
        # Копируем столбцы p и q
        col_p = np.empty(n, dtype=S.dtype)
        col_q = np.empty(n, dtype=S.dtype)
        for i in range(n):
            col_p[i] = S[k, i, p]
            col_q[i] = S[k, i, q]
        # Обновляем столбцы p и q
        for i in range(n):
            S[k, i, p] = cos_val * col_p[i] + sin_val * col_q[i]
            S[k, i, q] = -sin_val * col_p[i] + cos_val * col_q[i]

# Локальное обновление для V: обновляем только соответствующие столбцы
@numba.njit
def local_update_V(V, p, q, cos_val, sin_val):
    n = V.shape[0]
    for i in range(n):
        temp_p = V[i, p]
        temp_q = V[i, q]
        V[i, p] = cos_val * temp_p + sin_val * temp_q
        V[i, q] = -sin_val * temp_p + cos_val * temp_q

# JIT-компилированная функция для совместной диагонализации
@numba.njit
def joint_diagonalization_numba(S, eps, max_iter):
    '''
        Задача оптимизации решается с помощью ортогональных матриц Гивенса G(p, q, theta) (алгорит вращения Гивенса)
        P = П_m G_m(p_m, q_m, theta_m), где p_m != q_m 
        и P^T * S_k * P -> diag
    '''
    n_matrices, n, _ = S.shape
    P = np.eye(n) # Первое приближение - единичная матрица
    for it in range(max_iter):
        delta = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                g_num = 0.0
                g_den = 0.0
                for k in range(n_matrices):
                    s_pp = S[k, p, p]
                    s_qq = S[k, q, q]
                    s_pq = S[k, p, q]
                    g_num += 2 * s_pq * (s_pp - s_qq)
                    g_den += (s_pp - s_qq)**2 - 4 * s_pq**2
                theta = 0.5 * np.arctan2(g_num, g_den) # формула theta по всем матрицам k сразу
                if np.abs(theta) > eps:
                    cos_val = np.cos(theta)
                    sin_val = np.sin(theta)
                    local_update_V(P, p, q, cos_val, sin_val)
                    local_update(S, p, q, cos_val, sin_val)
                    if np.abs(theta) > delta:
                        delta = np.abs(theta)
        if delta < eps:
            print("Convergence reached: delta=" + str(delta) + " < eps=" + str(eps))
            break
    else:
        print("Stopped after" + str(max_iter) + "iterations without full convergence")
    return P, S

def compute_paLOSi_full(epochs, freq_min=0.1, freq_max=40, eps=1e-6, max_iter=1000):
    '''Наборами для CPC анализа будут являтся k спектров, соответсующей частоты
        Данные спектров S_k не центриуются по каналам.
        Ищем такую матрицу собственных векторов P общую для всех наборов данных, что sum_k offdiag(P^T * S_k * P)^2 -> min
        Так, D_k = P*S_k*P^T - матрица собственных значений
        '''
    # Извлекаем данные: (n_epochs, n_channels, n_times)
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    fs = epochs.info['sfreq']
    
    # Подсчёт спектра
    fft_data = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(n_times, d=1/fs)
    
    # Выделение области интереса частот для вычисления PaLOSi 
    freq_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    n_freq = len(freq_idx)
    fft_data_sel = fft_data[:, :, freq_idx]  # shape: (n_epochs, n_channels, n_freq)
    
    # Вычисление ковариационных матриц
    S_all = np.einsum('enc,ens->ncs', fft_data_sel, fft_data_sel.conj()) / n_epochs
    S_all = S_all.real  # из-за числовых погрешностей при перемножение матриц с комплексно-сопряжёнными берём Real, хотя должны быть только реальными
    
    # Решение задачи совместной диагонализации (поиск матрицы P-собственные вектора и соответсвтующих матриц D_k-собственных значений для каждого набора)
    P, D = joint_diagonalization_numba(S_all, eps=eps, max_iter=max_iter)
    
    # PaLOSi = (∑_f max(diag(D_f))) / (∑_f tr(D_f))
    diag_vals = np.diagonal(D, axis1=1, axis2=2)
    numerator = np.sum(np.max(diag_vals, axis=1))
    denominator = np.sum(diag_vals)
    PaLOSi = numerator / denominator if denominator != 0 else np.nan
    error = np.sqrt(2) * (eps + 1/max_iter)
    return PaLOSi, error


# Example
# max_iter = 3000
# eps = 1e-8
# n_jobs= -1
# epochs = np.rand_like((5, 60, 2000))
# palosi, d_palosi = compute_paLOSi_full(epochs, freq_min=0.1, freq_max=45, eps=eps, max_iter=max_iter)