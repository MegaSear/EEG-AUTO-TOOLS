# Copyright 2025 Sear Gamemode
import os 
import mne
import time
import scipy
import numpy as np 
import pickle

import matplotlib.pyplot as plt 

from tqdm import tqdm
from itertools import chain
from mne.preprocessing import ICA
from sklearn.decomposition import PCA         
from collections import Counter
from scipy.signal import savgol_filter, medfilt, wiener
from mne_icalabel import label_components
import inspect

from .savers import compared_snr
from .quality_check import detect_bad_channels, compared_spectrum, compute_bad_epochs, set_montage, search_bridge_cluster
from .scenarious import preprocessing_events
from .metrics import calculate_SN_ratio
from .craft_events import make_ANT_events, make_RiTi_events, make_CB_events

MODULE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class Transform:
    DESCRIPTION = '''
    Base Block
    '''
    PARAMETERS = {"report": {"type": bool, "default": True}}
    
    def __init__(self, **kwargs):
        self.apply_parameters(**kwargs)
        self.repo_images = {}
        self.repo_data = {}
        
    def apply_parameters(self, **kwargs):
        for param_name in kwargs:
            if param_name not in self.PARAMETERS:
                raise ValueError(f"Unknown parameter '{param_name}' for {self.__class__.__name__}")

        for param_name, param_info in self.PARAMETERS.items():
            value = kwargs.get(param_name, param_info["default"])
            setattr(self, param_name, value)

    def __call__(self, inst):
        return self.fit(inst.copy())

    def fit(self, inst):
        if self.report:
            self.repo_images = {}
            self.repo_data = {}
        return inst
    
    def save_report(self, path, pref=''):
        pathes = []
        for key, fig in self.repo_images.items():
            path_image = os.path.join(path, pref+key+'.png')
            fig.savefig(path_image, bbox_inches='tight')
            pathes.append(path_image)
        return pathes
    
    def get_report(self):
        return self.repo_data, self.repo_images
    
    @classmethod
    def get_parameters_schema(cls):
        return cls.PARAMETERS

class Sequence:
    def __init__(self, **transforms):
        self.transforms = transforms
        self.seq_report = []
        self.insts = []
        for name, transform in transforms.items():
            setattr(self, name, transform)

    def __call__(self, raw, progress_bar=None, cash=False):
        for name, transform in self.transforms.items():
            if progress_bar:
                progress_bar.set_postfix(status=f'{name}')
            raw = transform(raw)
            if cash:
                self.insts.append(raw)
        return raw
    
    def get_reports(self):
        reports = {}
        for name, transform in self.transforms.items():
            reports[name] = transform.get_report()
        return reports 
    
class ART(Transform):
    DESCRIPTION = '''
    Artifact Removal Transformer (ART)
        A neural network trained to clean the eeg
    '''

    PARAMETERS = {
        "window_size": {"type": int, "default": 2048},
        "overlap": {"type": float, "default": 0.2},
        "device": {"type": "choice", "default": "cpu", "choices": ["cpu", "cuda"]},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)
        self.model_path = os.path.join(MODULE_DIR, "cleaning_models", "ART.onnx")
        self.session = None
        self.model_c_max = None 

    def _lazy_load_session(self):
        if self.session is None:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            input_shape = self.session.get_inputs()[0].shape
            self.model_c_max = input_shape[1]

    def _clean_array(self, data):
        cleaned = np.zeros_like(data)
        counts = np.zeros_like(data)

        n_channels, n_times = data.shape
        step = int(self.window_size * (1 - self.overlap))
        for start in range(0, n_times, step):
            end = start + self.window_size
            if end > n_times:
                end = n_times
                start = max(0, end - self.window_size)

            window = data[:, start:end]  # (n_channels, window_size)
            if window.shape[1] < self.window_size:
                pad_width = self.window_size - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad_width)), mode='constant')

            pad_c = self.model_c_max - window.shape[0]
            if pad_c > 0:
                window = np.pad(window, ((0, pad_c), (0, 0)), mode='constant')

            window_tensor = window[np.newaxis, :, :]  # (1, n_channels, window_size)
            window_tensor = window_tensor.astype(np.float32)

            output = self.session.run(None, {'input': window_tensor})[0]  # (1, window_size, n_channels)
            output = np.transpose(output[0], (1, 0))  # (n_channels, window_size)
            
            output = output[:n_channels, :]
            valid_len = min(window.shape[1], end - start)
            cleaned[:, start:start+valid_len] += output[:, :valid_len]
            counts[:, start:start+valid_len] += 1

        cleaned = cleaned / np.maximum(counts, 1)
        return cleaned

    def fit(self, inst):
        self._lazy_load_session()
        if isinstance(inst, mne.io.BaseRaw):
            raw = inst.copy()
            data = raw.get_data()
            cleaned = self._clean_array(data)
            raw._data[:] = cleaned
            if self.report:
                self.repo_data = {}
            return raw

        elif isinstance(inst, mne.Epochs):
            epochs = inst.copy()
            data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            cleaned_data = np.zeros_like(data)

            for i in range(data.shape[0]):
                cleaned_data[i] = self._clean_array(data[i])

            epochs._data[:] = cleaned_data
            if self.report:
                self.repo_data = {}
            return epochs

        else:
            raise TypeError("Input must be Raw or Epochs!")


class RenameChannels(Transform):
    DESCRIPTION = '''
    Changes channel names
    '''
    PARAMETERS = {
        "channel_mapping": {"type": dict, "default": None},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, inst):
        if not self.channel_mapping:
            return inst 
        
        raw = inst.copy()
        current_channels = raw.ch_names
        rename_dict = {}
        
        for old_name, new_name in self.channel_mapping.items():
            if old_name in current_channels:
                rename_dict[old_name] = new_name
            else:
                print(f"⚠ Канал {old_name} отсутствует в данных, пропускается.")

        if not rename_dict:
            print("⚠ Нет подходящих каналов для переименования.")
            return raw

        raw.rename_channels(rename_dict)

        if self.report:
            self.repo_data = {}
            self.repo_images = {}
        return raw
    
class FrequencyFilter(Transform):
    DESCRIPTION = '''
    Notch Filter:
        notch_freq is None: apply filter cutter
        notch_freq is not None: not apply a filter cutter

    Range Filter:
        l_freq < h_freq: band-pass filter
        l_freq > h_freq: band-stop filter
        l_freq is not None and h_freq is None: high-pass filter
        l_freq is None and h_freq is not None: low-pass filter
        l_freq is None and h_freq is None: all-pass
    '''
    PARAMETERS = {
        "l_freq": {"type": float, "default": 0.1},
        "h_freq": {"type": float, "default": 45.0},
        "notch_freq": {"type": float, "default": None},
        "fir_design": {"type": "choice", "default": "firwin", "choices": ["firwin", "firwin2"]},
        "iir_params": {"type": dict, "default": {"order": 4, "ftype": "butter"}},
        "method": {"type": "choice", "default": 'fir', "choices": ['fir', 'iir']},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        raw_filtered = raw.copy()
        n_jobs = 'cuda' if mne.cuda.get_config()['MNE_USE_CUDA'] == 'true' else -1
        if self.notch_freq:
            raw_filtered = raw_filtered.notch_filter(
                            freqs=self.notch_freq, 
                            method=self.method, 
                            fir_design=self.fir_design if self.method == 'fir' else None,
                            iir_params=self.iir_params if self.method == 'iir' else None,
                            pad='reflect_limited', phase='zero-double', verbose=False, 
                            n_jobs=n_jobs)
            
        raw_filtered = raw_filtered.filter(
                        l_freq=self.l_freq, h_freq=self.h_freq, 
                        method=self.method, 
                        fir_design=self.fir_design if self.method == 'fir' else None, 
                        iir_params=self.iir_params if self.method == 'iir' else None, 
                        pad='reflect_limited', phase='zero-double', verbose=False, 
                        n_jobs=n_jobs)
        if self.report:
            fig = compared_spectrum(raw, raw_filtered, fmin=0, fmax=min(100, raw.info['sfreq']//2))
            self.repo_images = {'Spectrum': fig}
            self.repo_data = {}
        return raw_filtered


class PCAEpochs(Transform):
    DESCRIPTION = '''
    Not automatic transformation (for now)
        Allows Principal Component Analysis
        Recommended for viewing via reports
    '''

    PARAMETERS = {
        "whiten": {"type": bool, "default": False},
        "n_components": {"type": (int, float, str), "default": None},
        "random_state": {"type": int, "default": None},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, epochs):
        self.epochs = epochs
        data = epochs.get_data()  
        self.data_reshaped = data.transpose(1, 0, 2).reshape(data.shape[1], -1).T
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        self.pca.fit(self.data_reshaped) 
        self.info = epochs.info
        self.repo_images = {'PCA': self.plot()}

    def plot(self, n_components_to_plot=20):
        print(f"Explained variance: {self.pca.explained_variance_ratio_}")
        fig, axes = plt.subplots(2, n_components_to_plot // 2 + 1, figsize=(15, 6))
        axes = axes.flatten()
        for i in range(n_components_to_plot):
            component_map = self.pca.components_[i][:self.info['nchan']]  # Берем только EEG-каналы
            mne.viz.plot_topomap(component_map, self.info, axes=axes[i], show=False)
            axes[i].set_title(f"Component {i}")
        plt.tight_layout()
        plt.show()
        return fig

    def apply(self, components_to_keep):
        data_pca = self.pca.transform(self.data_reshaped)[:, components_to_keep]
        data_reconstructed = data_pca @ self.pca.components_[components_to_keep]
        data_reconstructed = data_reconstructed.T.reshape(self.n_chan, self.epochs.get_data().shape[0], -1).transpose(1, 0, 2)
        epochs_clean = self.epochs.copy()
        epochs_clean._data = data_reconstructed
        if not epochs_clean._data.flags['C_CONTIGUOUS']:
            epochs_clean._data = np.ascontiguousarray(epochs_clean._data)
        return epochs_clean
    

class ChannelSelector(Transform):
    DESCRIPTION = '''
    For EEG analysis, remove unnecessary channels, 
        such as those associated with other modalities
    '''

    PARAMETERS = {
        "exclude": {"type": list, "default": ["EOG", "BIP1"]},
        "include": {"type": list, "default": None},
        "eeg": {"type": bool, "default": True},
        "meg": {"type": bool, "default": False},
        "stim": {"type": bool, "default": False},
        "eog": {"type": bool, "default": False},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        if self.include:
            included = set([ch for ch in raw.ch_names if ch in self.include])
            raw = raw.pick_channels(included)
        if self.exclude:
            dropped = set([ch for ch in raw.ch_names if ch in self.exclude])
            raw = raw.drop_channels(dropped)
        picks_eeg = mne.pick_types(raw.info, eeg=self.eeg, meg=self.meg, stim=self.stim, eog=self.eog)
        raw = raw.pick(picks_eeg)
        return raw


class DetrendEpochs(Transform):
    DESCRIPTION = '''
    Applies detrending to EEG epochs individually
    '''
    PARAMETERS = {
        "detrend_type": {"type": "choice", "default": 'linear', "choices": ['constant', 'linear']},
        "report": {"type": bool, "default": True}
    }
    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, epochs):
        detrended_epochs = epochs.copy().apply_function(scipy.signal.detrend, type=self.detrend_type)
        if self.report:
            self.repo_data = {}
            snr_matrix = calculate_SN_ratio(epochs.copy(), option='mean_epochs', mode='log')
            snr_matrix_after = calculate_SN_ratio(detrended_epochs.copy(), option='mean_epochs', mode='log')
            snr_dist_fig = compared_snr([snr_matrix, snr_matrix_after], ['Before detrend', 'After detrend'])
            spectrum_fig = compared_spectrum(epochs, detrended_epochs, fmin=0, fmax=min(100, epochs.info['sfreq']))
            self.repo_images = {'SNR_dist': snr_dist_fig, 'Spectrum': spectrum_fig}
        return detrended_epochs


class BaselineEpochs(Transform):
    DESCRIPTION = '''
    Applies baselining to EEG epochs individually
    '''
    PARAMETERS = {
        "baseline": {"type": tuple, "default": (0, None)},
        "report": {"type": bool, "default": True}
    }
    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, epochs):
        baselined_epochs = epochs.copy().apply_baseline(self.baseline, verbose=False)
        if self.report:
            self.repo_data = {}
            snr_matrix = calculate_SN_ratio(epochs.copy(), option='mean_epochs', mode='log')
            snr_matrix_after = calculate_SN_ratio(baselined_epochs.copy(), option='mean_epochs', mode='log')
            snr_dist_fig = compared_snr([snr_matrix, snr_matrix_after], ['Before baseline', 'After baseline'])
            spectrum_fig = compared_spectrum(epochs, baselined_epochs, fmin=0, fmax=min(100, epochs.info['sfreq']/2))
            self.repo_images = {'SNR_dist': snr_dist_fig, 'Spectrum': spectrum_fig}
        return baselined_epochs
    

class BadEpochsDetector(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "roi_channels": {"type": list, "default": None},
        "apply": {"type": bool, "default": True},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, epochs):
        snr_matrix = calculate_SN_ratio(epochs.copy(), option='mean_epochs', mode='log')
        rej_dict = compute_bad_epochs(epochs, snr_matrix, roi_channels=self.roi_channels)
        cleaned_epochs = epochs.copy().drop(rej_dict['FINAL'], verbose=False)
        if self.report:
            snr_matrix_after = calculate_SN_ratio(cleaned_epochs, option='mean_epochs', mode='log')
            snr_dist_fig = compared_snr([snr_matrix, snr_matrix_after], ['Before detector', 'After Detector'])
            spectrum_fig = compared_spectrum(epochs, cleaned_epochs, fmin=0, fmax=min(100, epochs.info['sfreq']/2))
            self.repo_images = {'SNR_dist': snr_dist_fig, 'Spectrum': spectrum_fig}
            self.repo_data = rej_dict
        if self.apply:
            return cleaned_epochs
        else:
            return epochs


class CheckEvents(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "scenarious_name": {"type": str, "default": None},
        "report": {"type": bool, "default": True}
    }
    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        new_events, reverse_event_id, repo_data = preprocessing_events(raw, scenarious=self.scenarious_name)
        if self.report:
            self.repo_data = repo_data
            self.repo_images = {}
        event_times = new_events[:, 0] / raw.info['sfreq']
        event_times -= raw.first_time
        event_codes = new_events[:, 2].astype(int)
        event_descriptions = [reverse_event_id[code] for code in event_codes]
        durations = [0] * len(event_times)
        new_annotations = mne.Annotations(onset=event_times,
                                        duration=durations,
                                        description=event_descriptions,
                                        )
        raw.set_annotations(new_annotations)
        return raw


class Cropping(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "stimulus": {"type": list, "default": None},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs): 
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        events, event_id = mne.events_from_annotations(raw)
        if event_id is None or not self.stimulus:
            return raw
        stimulus_codes = [event_id[stim] for stim in self.stimulus if stim in event_id]
        desired_event_ids = dict(zip(self.stimulus, stimulus_codes))
        reverse_desired_event_ids = dict(zip(stimulus_codes, self.stimulus))
        desired_event_values = list(desired_event_ids.values())
        filtered_events = events[np.isin(events[:, 2], desired_event_values)]
        filtered_events = np.array(filtered_events)
        annotations = mne.annotations_from_events(
            events=filtered_events,
            sfreq=raw.info['sfreq'],
            event_desc=reverse_desired_event_ids,
            first_samp=raw.first_samp
        )
        raw.set_annotations(annotations)
        first_stimulus_time = filtered_events[0, 0] / raw.info['sfreq']
        last_stimulus_time = filtered_events[-1, 0] / raw.info['sfreq']
        tmin = max(0, first_stimulus_time - 1.0)
        tmax = min(raw.times[-1], last_stimulus_time + 1.0)
        raw = raw.crop(tmin=tmin, tmax=tmax)
        if self.report:
            missing_labels = set(self.stimulus) - set(event_id.keys())
            self.repo_data = {'Deleted events': missing_labels, 'New_duration': tmax-tmin}
            self.repo_images = {}
        return raw

class StatisticFilter(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "method": {"type": "choice", "default": 'savgol', "choices": ['savgol', 'medfilt', 'wiener', 'moving_avg', 'robust']},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs): 
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        raw_filtered = raw.copy()
        match self.method:    
            case 'savgol':
                data = raw.get_data()
                for idx, ch in tqdm(enumerate(raw.ch_names), total=len(raw.ch_names)):
                    data[idx] = savgol_filter(data[idx], window_length=20, polyorder=4)
                raw_filtered._data = data

            case 'medfilt':
                kernel_size = 11  # Must be odd, adjust based on sampling rate
                for idx, ch in tqdm(enumerate(raw.ch_names), total=len(raw.ch_names)):
                    data[idx] = medfilt(data[idx], kernel_size=kernel_size)

            case 'wiener':
                mysize = 15  # Window size for local statistics, adjust as needed
                for idx, ch in tqdm(enumerate(raw.ch_names), total=len(raw.ch_names)):
                    data[idx] = wiener(data[idx], mysize=mysize)

            case 'moving_avg':
                window_size = 10  # Adjust based on sampling rate
                window = np.ones(window_size) / window_size
                for idx, ch in tqdm(enumerate(raw.ch_names), total=len(raw.ch_names)):
                    data[idx] = np.convolve(data[idx], window, mode='same')

            case 'robust':
                for idx, ch in tqdm(enumerate(raw.ch_names), total=len(raw.ch_names)):
                    signal = data[idx]
                    # Compute MAD
                    median = np.median(signal)
                    mad = np.median(np.abs(signal - median))
                    threshold = 3 * mad  # Threshold for outlier detection
                    # Replace outliers with median
                    outliers = np.abs(signal - median) > threshold
                    signal[outliers] = median
                    data[idx] = signal

            case _:
                raise f"Unknown method: {self.method}"
            
        if self.report:
            fig = compared_spectrum(raw, raw_filtered, fmin=0, fmax=100)
            self.repo_images = {'Spectrum': fig}
            self.repo_data = {}
        return raw_filtered
    

class Interpolate(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "reset_bads": {"type": bool, "default": True},
        "eeg_method": {"type": "choice", "default": "spline", "choices": ["spline"]},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs): 
        super().__init__()
        self.apply_parameters(**kwargs)
        
    def fit(self, raw):
        cleaned_raw = raw.copy().interpolate_bads(reset_bads=self.reset_bads, method=dict(eeg=self.eeg_method), verbose=False)
        if self.report:
            fig = compared_spectrum(raw, cleaned_raw, fmin=0, fmax=100)
            self.repo_images = {'Spectrum': fig}
            self.repo_data = {}
        return cleaned_raw


class BadChannelsDetector(Transform):
    DESCRIPTION = '''
    Allows detection of channels:
        Low amplitude
        High amplitude
        Bridging
        Poorly Reconstructed
    Marks them as bads (This may affect some Transformations)
    '''
    PARAMETERS = {
        "method_noise": {"type": "choice", "default": "auto", "choices": ["ransac", "snr", "lof", "ed", "corr", "auto"]},
        "method_bridge": {"type": "choice", "default": "auto", "choices": ["ed", "corr", "auto"]},
        "n_jobs": {"type": int, "default": 3},
        "mark": {"type": bool, "default": True},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        self.electrodesD, clusters, bridge_figs, noised_fig = detect_bad_channels(raw, self.method_noise, self.method_bridge, self.n_jobs)
        self.bad_channels = [ch for sublist in self.electrodesD.values() for ch in sublist]
        united_bad_channels = list(set(self.bad_channels) | set(raw.info['bads']))
        if self.mark:
            raw.info['bads'] = united_bad_channels

        if self.report:
            self.repo_images = {'Bridged_channels': bridge_figs[0], 'Bridged_hist': bridge_figs[1], 
                                'Noised_channels': noised_fig}
            self.repo_data = {**{'Clusters': clusters}, **self.electrodesD, **{'FINAL': united_bad_channels}}
            self.repo_data['N_HighAmp'] =  len(self.repo_data['HighAmp'])
            self.repo_data['N_LowAmp'] = len(self.repo_data['LowAmp'])
            self.repo_data['N_Bridged'] = len(self.repo_data['Bridged'])
            self.repo_data['N_Noise_Rate'] = len(self.repo_data['Noise_Rate'])
        return raw

class AutoICA(Transform):
    DESCRIPTION = '''
    Applies independent component analysis
        Allows to automatically calculate the number of components
        With IClabel (neural network) allows to automatically mark independent components and remove them.
    '''
    PARAMETERS = {
        "auto_n_components": {"type": bool, "default": True},
        "n_components": {"type": (int, float), "default": 0.999999},
        "method": {"type": "choice", "default": 'infomax-extended', "choices": ['fastica', 'infomax', 'infomax-extended']},
        "auto_max_deleted": {"type": bool, "default": True},
        "delete_ratio": {"type": float, "default": 1.0},
        "brain_conf_ratio": {"type": float, "default": 0},
        "exclude_bads": {"type": bool, "default": True},
        "report": {"type": bool, "default": True}
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs) 

    def fit(self, raw):
        if self.auto_n_components:
            rank = len(raw.ch_names)
            if raw.info['custom_ref_applied']:
                rank -= 1 
            if self.exclude_bads:
                rank -= len(raw.info['bads'])
            self.n_components=max(1, rank)

        if self.exclude_bads:
            picks = mne.pick_channels(ch_names=raw.info['ch_names'], include=[], exclude=raw.info["bads"])
        else:
            picks = mne.pick_channels(ch_names=raw.info['ch_names'], include=[])
        
        match self.method:
            case 'infomax':
                fit_params = dict(ortho=False, extended=False)
            case 'infomax-extended':
                fit_params = dict(ortho=False, extended=True)
            case 'fastica':
                fit_params = dict(ortho=True, extended=True)

        ica = ICA(n_components=self.n_components, fit_params=fit_params,
                method='picard', random_state=42, verbose=False)
        
        ica.fit(raw, picks=picks, verbose=False)
        
        mne.set_log_level('ERROR') 
        ica_labels = label_components(raw, ica, method='iclabel')

        mne.set_log_level('INFO')
        labels, probas = ica_labels["labels"], ica_labels['y_pred_proba']
        
        ica_sources = ica.get_sources(raw.copy())
        self.sources_data = ica_sources.get_data()
        self.labels = labels
        self.probas = probas 

        cond = lambda label, proba: label not in ['brain'] or (label in ['brain'] and proba <= self.brain_conf_ratio)
            
        exclude_idx    = [idx   for idx, (label, proba) in enumerate(zip(labels, probas)) if cond(label, proba)]
        exclude_labels = [label for idx, (label, proba) in enumerate(zip(labels, probas)) if cond(label, proba)]
        
        if self.auto_max_deleted:
            MAX_DELETED_COMP = ica.n_components_//2
        else:
            MAX_DELETED_COMP = int(len(exclude_idx) * self.delete_ratio)

        exclude_idx = exclude_idx[:MAX_DELETED_COMP]
        exclude_labels = exclude_labels[:MAX_DELETED_COMP]

        raw_for_apply = raw.copy().pick(picks)
        raw_for_apply.info['bads'] = []
        raw_filtered_partial = ica.apply(raw_for_apply, exclude=exclude_idx, verbose=False)

        raw_filtered = raw.copy()
        raw_filtered._data[picks, :] = raw_filtered_partial._data

        self.repo_images = {}

        if self.report:
            all_components_fig = ica.plot_components(inst=raw, show=False, verbose=False)
            if isinstance(all_components_fig, list):
                for fig in all_components_fig:
                    plt.close(fig)
            else:
                plt.close(all_components_fig)

            for idx, label in zip(exclude_idx, exclude_labels):
                fig = ica.plot_properties(raw, picks=idx, show=False, verbose=False)[0]
                self.repo_images[f'comp_{idx}_label_{label}'] = fig
                plt.close(fig)
                
            self.repo_images['all_comp'] = all_components_fig 
            spec_fig = compared_spectrum(raw.copy().pick(picks), raw_filtered.copy().pick(picks), 
                                        fmin=0, fmax=min(100, raw.info['sfreq']/2))
            self.repo_images['Spectrum'] = spec_fig
            self.repo_data = {'exclude_idx': exclude_idx, 'exclude_labels': exclude_labels, 
                              'N_exclude_idx': len(exclude_idx)}

        return raw_filtered


class Rereference(Transform): 
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "method": {"type": "choice", "default": 'average', "choices": ['average', 'laplas', 'linked_mastoid', 'custom']},   
        "ref_channels": {"type": list, "default": None}, 
        "exclude_bads": {"type": bool, "default": False},
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        raw_filtered = raw.copy()
        picks = mne.pick_channels(ch_names=raw.info['ch_names'], include=[], exclude=raw.info["bads"])

        match self.method:
            case 'average':
                if self.exclude_bads:
                    include_channels = [ch for ch in raw.ch_names if ch not in raw.info['bads']]
                    raw_filtered.set_eeg_reference(ref_channels=include_channels, projection=False, verbose=False)
                else:
                    raw_filtered.set_eeg_reference(self.method, verbose=False)
            case 'laplas':
                raw_filtered = mne.preprocessing.compute_current_source_density(raw_filtered, verbose=False)
            case 'linked_mastoid':
                raw_filtered.set_eeg_reference(ref_channels=['M1', 'M2'], projection=False, verbose=False)
            case 'custom':
                raw_filtered.set_eeg_reference(ref_channels=self.ref_channels, projection=False, verbose=False)
        raw_filtered.set_eeg_reference(ref_channels=[], projection=False, verbose=False)
        if self.report:
            figs = compared_spectrum(raw.copy().pick(picks), raw_filtered.copy().pick(picks), fmin=0, fmax=100)
            self.repo_data = {}
            self.repo_images = {'Spectrum': figs}

        return raw_filtered


class Resample(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "sfreq": {"type": float, "default": 128},   
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        resampled_raw = raw.copy().resample(sfreq=self.sfreq, npad="auto")
        if self.report:
            self.repo_data = {}
            self.repo_images = {}
        return resampled_raw
    

class BridgeInterpolate(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "bridged_idx": {"type": list, "default": None},   
        "method_bridge": {"type": "choice", "default": 'ed', "choices": ["corr", "ed", "auto"]},  
        "reset_bridges": {"type": bool, "default": True}, 
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        ch_names = raw.ch_names
        if self.bridged_idx is None:
            clusters, bridge_figs, bridged_idx = search_bridge_cluster(raw, method=self.method_bridge)
            self.bridged_idx = bridged_idx
            bridged_electrodes = list(set(list(chain.from_iterable(clusters))))
            self.repo_images = {'Bridged_channels': bridge_figs[0], 'Bridged_hist': bridge_figs[1]}
            self.repo_data = {'Clusters': clusters, "Bridged_Electrodes": bridged_electrodes}
        else:
            unique_idxs = {idx for pair in self.bridged_idx for idx in pair}
            bridged_electrodes = [ch_names[i] for i in unique_idxs]

        mne.set_log_level("ERROR")
        raw = mne.preprocessing.interpolate_bridged_electrodes(raw, self.bridged_idx, bad_limit=len(ch_names))
        mne.set_log_level("WARNING")

        if self.reset_bridges:
            raw.info['bads'] = [ch for ch in raw.info['bads'] if ch not in bridged_electrodes]
        return raw


class Raw2Epoch(Transform):
    DESCRIPTION = '''
    
    '''
    PARAMETERS = {
        "tmin": {"type": float, "default": -0.15},   
        "tmax": {"type": float, "default": 0.75},   
        "baseline": {"type": tuple, "default": (None, 0)},   
        "stimulus_list": {"type": "choice", "default": 'ed', "choices": ["corr", "ed", "auto"]},  
        "scenarious_name": {"type": "choice", "default": None, "choices": [None, "RiTi", "ANT", "MAIN", "Test-IAT"]}, 
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)

    def fit(self, raw):
        if self.scenarious_name == 'RiTi':
            events, event_id = make_RiTi_events(raw, self.stimulus_list)
        elif self.scenarious_name == 'ANT':
            events, event_id = make_ANT_events(raw, self.tmin, self.tmax, self.baseline)
        elif self.scenarious_name == 'Rest-IAT' or self.scenarious_name=="MAIN":
            events, event_id = make_CB_events(raw, self.scenarious_name)
        else:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=self.tmin, tmax=self.tmax, 
                        baseline=self.baseline, preload=True, verbose=False, event_repeated="merge")

        if self.report:
            events = epochs.events
            event_id = epochs.event_id
            event_counts = Counter(events[:, 2])
            event_id_reverse = {v: k for k, v in event_id.items()}
            dict_event_count = {}

            for event_code, count in event_counts.items():
                dict_event_count[event_id_reverse[event_code]] = count
            self.repo_data = {'events_count': dict_event_count}
        return epochs


class SetMontage(Transform):
    DESCRIPTION = '''
    Applies montage to EEG (this affects the visual component of the reports AND interpolation)
    '''
    PARAMETERS = {
        "montage": {"type": "choice", "default": "waveguard64", "choices": mne.channels.get_builtin_montages() + ["waveguard64", "personal"]},   
        "elc_file": {"type": "str", "default": None},
        "mode": {"type": str, "default": "Cz"},   
        "threshold": {"type": float, "default": 0.08},  
        "interpolate": {"type": bool, "default": True}, 
        "report": {"type": bool, "default": True}
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.apply_parameters(**kwargs)
        
    def fit(self, raw):
        return set_montage(raw, self.montage, self.mode, self.threshold, self.interpolate)

