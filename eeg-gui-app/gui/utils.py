import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import os

from PyQt5.QtCore import QThread, pyqtSignal, QObject
import os
import mne
import numpy as np
import pandas as pd
import pickle
import sys
import h5py
import json
import hashlib

# Путь к библиотеке eeg_auto_tools
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, lib_path)
from eeg_auto_tools.transforms import Transform
import os
from pathlib import Path


class Worker(QObject):    
    progress = pyqtSignal(str)
    log = pyqtSignal(dict)
    finished = pyqtSignal(dict, dict, dict)
    error = pyqtSignal(str)

    def __init__(self, graph_manager, cache_dir):
        super().__init__()
        self.graph_manager = graph_manager
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def run(self):
        from joblib import parallel_config
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with parallel_config(
                    temp_folder=tmpdir,
                    max_nbytes=0,           
                    backend="loky"):
                if not hasattr(self.graph_manager, 'input_files'):
                    self.error.emit("⚠ Файлы для обработки не заданы.")
                    return

                graph, reverse_graph = self.graph_manager._build_graph()
                self.graph_manager._check_connectivity(graph)
                self.graph_manager._check_no_cycles(graph)

                transform_names = self.graph_manager._assign_transform_names(graph)
                paths = self.graph_manager._find_all_paths(graph)
                self.progress.emit(f"🧩 Найдено {len(paths)} путей для обработки.")

                reports_per_file = {}
                logs = {}

                for file_idx, file in enumerate(self.graph_manager.input_files):
                    self.progress.emit(f"⚙ Обработка файла: {file} ({file_idx + 1}/{len(self.graph_manager.input_files)})")
                    raw = mne.io.read_raw(file, preload=True, verbose=False)
                    logs.setdefault(file, {})
                    file_report = {}
                    
                    def process_node(node, t_name, current_raw):
                        try:
                            transform = node.transform_class(**node.params)
                        except Exception as e:
                            raise RuntimeError(f"Ошибка создания Transform {t_name}: {e}")
                        try:
                            processed = transform(current_raw)
                        except Exception as e:
                            raise RuntimeError(f"Ошибка выполнения Transform {t_name}: {e}")
                        try: 
                            repo_data, repo_images = transform.get_report()
                        except Exception as e:
                            raise RuntimeError(f"Ошибка получения отчета Transform {t_name}: {e}")
                        return processed, repo_data, repo_images
                    
                    def update_report(t_name, repo_data, repo_images):
                        try:
                            if repo_data:
                                if t_name not in file_report:
                                    file_report[t_name] = {}
                                file_report[t_name].update(repo_data)
                            else:
                                if repo_images:
                                    if t_name not in file_report:
                                        file_report[t_name] = {}
                                    file_report[t_name].update(repo_images)
                        except Exception as e:
                            raise RuntimeError(f"Ошибка обновления репортов для {t_name}: {e}")

                    def compute_hash(nodes, file_path):
                        node_descriptions = [(n.name, n.params) for n in nodes]
                        return hashlib.md5(json.dumps((file_path, node_descriptions), sort_keys=True).encode()).hexdigest()
                    
                    def update_log(file, path_id, node, status):
                        logs[file].setdefault(path_id, {})
                        logs[file][path_id][node.name] = {
                            "params": dict(node.params),
                            "status": status
                        }
                        self.log.emit({
                            "file": file,
                            "path_id": path_id,
                            "node": node.name,
                            "params": dict(node.params),
                            "status": status
                        })
                        
                    for path_idx, path in enumerate(paths):
                        logs[file][path_idx] = {}
                        current_raw = raw
                        for node in path:
                            update_log(file, path_idx, node, "Waiting")
                        for node_idx, node in enumerate(path):
                            update_log(file, path_idx, node, "Computing")
                            hash_value = compute_hash(path[:node_idx+1], file)
                            try:
                                cache_base = os.path.join(self.cache_dir, f"{os.path.basename(file)}_{hash_value}_{node.name}.h5")
                                if os.path.exists(cache_base):
                                    update_log(file, path_idx, node, "Cache Reading")
                                    current_raw = self._load_cached_raw(cache_base)
                                    cached_repo_data, cached_repo_images = self._load_cached_report(file, hash_value, node.name)
                                    update_report(node.name, cached_repo_data, cached_repo_images)
                                    update_log(file, path_idx, node, "Cache Readed")
                                    if current_raw:
                                        continue
                                processed, repo_data, repo_images = process_node(node, node.name, current_raw)
                                current_raw = processed
                                update_report(node.name, repo_data, repo_images)
                                update_log(file, path_idx, node, "Caching")
                                self._cache_raw(processed, file, hash_value, node.name, node.params)
                                self._cache_report(file, hash_value, node.name, repo_data, repo_images)
                                update_log(file, path_idx, node, "Computed")
                            except Exception as e:
                                update_log(file, path_idx, node, "Failed")
                                import traceback
                                self.error.emit(f"❌ Ошибка в Transform {node.name}: {e}\n{traceback.format_exc()}")
                                return
                    
                    reports_per_file[file] = file_report
                    self.progress.emit(f"✅ Обработка файла завершена: {file})")
                self.finished.emit(reports_per_file, transform_names, logs)

    def _cache_raw(self, raw, file_path, hash_value, node_name, params):
        cache_file = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}_{hash_value}_{node_name}.h5")
        try:
            with h5py.File(cache_file, 'w') as f:
                f.create_dataset('data', data=raw.get_data(), compression='gzip', compression_opts=9)
                ch_names_dt = h5py.string_dtype('utf-8')
                ch_names_dataset = f.create_dataset('ch_names', (len(raw.ch_names),), dtype=ch_names_dt)
                for i, ch_name in enumerate(raw.ch_names):
                    ch_names_dataset[i] = ch_name.encode('utf-8') if isinstance(ch_name, str) else str(ch_name).encode('utf-8')
                f.create_dataset('info', data=np.void(pickle.dumps(raw.info)))
                params_json = json.dumps(params, ensure_ascii=False)
                f.create_dataset('params', data=params_json.encode('utf-8'), dtype=ch_names_dt)
            return cache_file
        
        except Exception as e:
            print(f"Ошибка кэширования: {e}")
            return None
    
    def _load_cached_raw(self, cache_file):
        if not self._verify_cache(cache_file):
            print(f"Кэш повреждён или неполный: {cache_file}")
            return None
        try:
            with h5py.File(cache_file, 'r') as f:
                data = f['data'][:]
                info = pickle.loads(f['info'][()])
            return mne.io.RawArray(data, info, verbose=False)
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            return None

    def sanitize_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sanitize_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def _cache_report(self, file_path, hash_value, node_name, repo_data, repo_images):
        report_file = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}_{hash_value}_{node_name}_report.json")
        figures_dir = Path(self.cache_dir) / f"{os.path.basename(file_path)}_{hash_value}_{node_name}_figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        sanitized_data = self.sanitize_for_json(repo_data)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(sanitized_data, f, ensure_ascii=False, indent=2)

        for key, fig in repo_images.items():
            fig_path = figures_dir / f"{key}.pkl"
            with open(fig_path, 'wb') as f_out:
                pickle.dump(fig, f_out)

    def _load_cached_report(self, file_path, hash_value, node_name):
        report_file = os.path.join(self.cache_dir, f"{os.path.basename(file_path)}_{hash_value}_{node_name}_report.json")
        figures_dir = Path(self.cache_dir) / f"{os.path.basename(file_path)}_{hash_value}_{node_name}_figures"

        repo_data = {}
        repo_images = {}

        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                repo_data = json.load(f)

        if figures_dir.exists():
            for fig_file in figures_dir.glob("*.pkl"):
                key = fig_file.stem
                with open(fig_file, 'rb') as f_in:
                    repo_images[key] = pickle.load(f_in)

        return repo_data, repo_images

    def _verify_cache(self, cache_file):
        if not h5py.is_hdf5(cache_file):
            return False
        try:
            with h5py.File(cache_file, 'r') as f:
                return all(key in f for key in ('data', 'ch_names', 'info'))
        except Exception:
            return False
    
class EEGFileManager:    
    @staticmethod
    def read_file_info(file_path):
        """Читает метаданные EEG-файла."""
        try:
            raw = mne.io.read_raw(file_path, preload=False, verbose=False)
            return {
                "duration": f"{raw.times[-1]:.2f}",
                "sfreq": f"{raw.info['sfreq']:.2f}",
                "n_channels": str(raw.info['nchan']),
                "ch_names": raw.ch_names
            }
        except Exception:
            return {"duration": "Ошибка", "sfreq": "Ошибка", "n_channels": "Ошибка"}

    @staticmethod
    def import_bids(bids_root):
        """Импортирует файлы из BIDS-директории."""
        files = []
        subjects = get_entity_vals(bids_root, 'subject')
        tasks = get_entity_vals(bids_root, 'task')
        all_runs = get_entity_vals(bids_root, 'run') or [None]

        for subject in subjects:
            for task in tasks:
                for run in all_runs:
                    bids_path = BIDSPath(subject=subject, task=task, run=run, root=bids_root)
                    if os.path.exists(bids_path.fpath):
                        try:
                            raw = read_raw_bids(bids_path=bids_path, verbose=False)
                            files.append(raw.filenames[0])
                        except Exception as e:
                            print(f"Ошибка чтения {subject}, {task}, run={run}: {e}")
        return files

    @staticmethod
    def rename_channels(file_path, channel_mapping, save_path=None):
        try:
            raw = mne.io.read_raw(file_path, preload=True, verbose=False)
            current_channels = raw.ch_names
            rename_dict = {}

            # Проверяем, какие каналы можно переименовать
            for old_name, new_name in channel_mapping.items():
                if old_name in current_channels:
                    rename_dict[old_name] = new_name
                else:
                    print(f"⚠ Канал {old_name} отсутствует в файле {file_path}")

            if not rename_dict:
                print(f"⚠ Нет подходящих каналов для переименования в {file_path}")
                return raw

            # Переименовываем каналы
            raw.rename_channels(rename_dict)

            # Сохраняем файл, если указан save_path
            if save_path:
                raw.save(save_path, overwrite=True, verbose=False)

            return raw
        except Exception as e:
            raise RuntimeError(f"Не удалось переименовать каналы в файле {file_path}: {e}")