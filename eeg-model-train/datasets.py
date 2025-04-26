###### Класс датасета возваращающий по __getitem__ пару clear и noised ээг отрезка
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch 
from scipy.signal import resample
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mne 
import warnings
import pickle
from collections import defaultdict

warnings.filterwarnings(
    "ignore",
    message=r".*does not conform to MNE naming conventions.*",
    category=RuntimeWarning
)

class EEGArtifactDataset(Dataset):
    def record_validation(self, pid, sol_map, iclabel_map, target_sfreq):
        sol_file    = sol_map[pid]
        iclabel_file= iclabel_map[pid]

        # Чтение входных файлов и удаление warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ica = mne.preprocessing.read_ica(sol_file, verbose=False)

        with open(iclabel_file, "rb") as f:
            components_info = pickle.load(f)
        
        labels  = components_info['labels']
        probas  = components_info['probas']
        sources = components_info['sources'] # Source shape: n_components_ICA * n_times
        orig_sfreq   = components_info['raw_info']['sfreq']
        n_bads = len(components_info['raw_info']['bads'])
        n_channels = components_info['raw_info']['nchan']
        
        # Валидация по некорректным sources
        if sources.size == 0 or np.any(np.isnan(sources)) or np.any(np.isinf(sources)):
            return None
        # Валидация по количеству каналов
        if n_channels < 100: # Датасет рассчитан на 120 каналов примерно
            return None 
            
        # Валидация по длине continious raw после понижения частоты дискретизации
        n_times_orig = sources.shape[1]
        n_times_tgt  = int(n_times_orig * target_sfreq / orig_sfreq)
        if n_times_tgt < self.segment_length:
            return None
            
        # Валидация по количеству плохих каналов ~ 15%
        if n_bads / n_channels >= 0.15:
            return None
            
        # Понижения частоты дискретизации
        sources_resampled = resample(sources, n_times_tgt, axis=1)

        # Валидация по неудачному понижению частоты дискретизации
        if np.any(np.isnan(sources_resampled)) or np.any(np.isinf(sources_resampled)):
            return None

        # Подготовка компонент для формирования обучающей выборки
        non_brain_comps = { 
            "eye blink": [], "muscle artifact": [], "heart beat": [], "channel noise": [], "other": [], "line noise": []
        }
        brain_comps = []
        for idx, (label, proba) in enumerate(zip(labels, probas)):
            # Выбор только тех Brain компонент в которых модель уверена
            if label == "brain" and proba > self.brain_threshold:
                brain_comps.append(idx)
            # Выбор всех non-brain компонент
            elif label in non_brain_comps:
                non_brain_comps[label].append(idx)

        # Валидация по количеству Brain и Other компонент
        if len(non_brain_comps['other']) >= self.max_other_componnets:
            return None
        if len(brain_comps) <= self.min_brain_components:
            return None

        # Валидация по будущей нарезке на непересекающиеся окна
        n_total   = sources_resampled.shape[1]
        n_windows = n_total // self.segment_length
        if n_windows <= 0:
            return None
        
        return ica, sources_resampled, brain_comps, non_brain_comps, n_windows

    
    def __init__(self, sol_files, iclabel_files, segment_length=4000, brain_threshold=0.8, target_sfreq=256, 
                 n_jobs=-1, device='cpu', min_brain_components=0, max_other_componnets=30):

        # Валидация входных файлов с параллелизацией
        self.segment_length = segment_length
        self.brain_threshold = brain_threshold
        self.device = device
        self.min_brain_components = min_brain_components
        self.max_other_componnets = max_other_componnets
        tmp_results = []
        with tqdm_joblib(tqdm(total=len(iclabel_files))) as progress_bar:
            tmp_results = Parallel(n_jobs=n_jobs, timeout=None)(
                delayed(self.record_validation)(pid, sol_files, iclabel_files, target_sfreq) 
                for pid in range(len(iclabel_files))
            )
        tmp_results = [res for res in tmp_results if res is not None]
        print(f"К анализу было допущено {len(tmp_results)} файлов")

        # Формирование непересекающихся пар (record_id, split_idx)
        # Изменение типа хранения данных с помощью привязки по record_id filtered
        self.icas                  = []
        self.sources               = []
        self.window_indices        = []  
        self.brain_components_list = []
        self.non_brain_components_list = []
        for new_idx, (ica, sources_resampled, brain_components, non_brain_components, n_windows) in enumerate(tmp_results):
            self.icas.append(ica)
            self.sources.append(sources_resampled)
            self.brain_components_list.append(brain_components)
            self.non_brain_components_list.append(non_brain_components)
            n_total = sources_resampled.shape[1]
            for i in range(n_windows):
                start_idx = i * self.segment_length
                if start_idx + self.segment_length <= n_total:
                    self.window_indices.append((new_idx, start_idx))
        
        component_counts = defaultdict(list)
        for brain_comps, non_brain_comps in zip(self.brain_components_list, self.non_brain_components_list):
            component_counts['brain'].append(len(brain_comps))
            for comp_type, comps in non_brain_comps.items():
                component_counts[comp_type].append(len(comps))
        n_classes = len(component_counts)
        fig, axes = plt.subplots(nrows=n_classes, ncols=1, figsize=(10, 4 * n_classes))
        for ax, (comp_type, counts) in zip(axes, component_counts.items()):
            ax.hist(counts, bins=100, alpha=0.7)
            ax.set_title(f'Распределение количества компонент: {comp_type}')
            ax.set_xlabel('Количество компонент')
            ax.set_ylabel('Частота')
        plt.tight_layout()
        plt.show()

    
    def generate_segment_from_sources(self, segment_sources, ica, brain_components, non_brain_components, eps=1e-10):
        all_non_brain = [idx for cat, inds in non_brain_components.items() for idx in inds]
        if not all_non_brain:
            noisy_components = brain_components.copy()
        else:
            k = np.random.randint(0, len(all_non_brain)+1) # выбирается количество non-brain комопонент для формирования noisy
            sel = np.random.choice(all_non_brain, k, replace=False)
            noisy_components = brain_components + sel.tolist()

        n_ch  = ica.get_components().shape[0]
        n_tim = segment_sources.shape[1]

        # Обратная проекция brain комопнент в пространство каналов
        if brain_components:
            clean = ica.get_components()[:, brain_components] @ segment_sources[brain_components, :]
        else:
            clean = np.zeros((n_ch, n_tim))

        # Добавляем шум
        if noisy_components:
            noisy = ica.get_components()[:, noisy_components] @ segment_sources[noisy_components, :]
        else:
            noisy = clean.copy()

        if noisy.shape[1] == 0:
            return np.zeros_like(clean), np.zeros_like(noisy)

        # Нормализация с защитой от NaN и деления на ноль
        m = noisy.mean(axis=1, keepdims=True)
        s = noisy.std(axis=1, keepdims=True)
        s = np.maximum(s, eps)

        clean = (clean - m) / s
        noisy = (noisy - m) / s
        return clean, noisy

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        
        rec_idx, start = self.window_indices[idx]
        ica     = self.icas[rec_idx]
        sources = self.sources[rec_idx]
        brain   = self.brain_components_list[rec_idx]
        nonb    = self.non_brain_components_list[rec_idx]

        # По idx выбирается пара (clean, noisy) ээг с помощью back proj ICA sources
        seg = sources[:, start:start+self.segment_length]
        clean, noisy = self.generate_segment_from_sources(seg, ica, brain, nonb)
        return (torch.tensor(clean,  dtype=torch.float32).to(self.device),
                torch.tensor(noisy, dtype=torch.float32).to(self.device))

    