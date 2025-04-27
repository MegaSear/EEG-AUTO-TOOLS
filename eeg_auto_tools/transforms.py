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

from .savers import compared_snr
from .quality_check import detect_bad_channels, compared_spectrum, compute_bad_epochs, set_montage, search_bridge_cluster
from .scenarious import preprocessing_events
from .metrics import calculate_SN_ratio
from .craft_events import make_ANT_events, make_RiTi_events, make_CB_events

class Transform:
    def __init__(self, report=False):
        self.repo_images = {}
        self.repo_data = {}
        self.report = report

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
    

class PCAEpochs(Transform):
    def __init__(self, n_components=None, whiten=False):
        self.whiten = whiten
        self.n_components = n_components
        self.repo_data = {}
        self.repo_images = {}

    def fit(self, epochs):
        self.epochs = epochs
        data = epochs.get_data()  
        self.data_reshaped = data.transpose(1, 0, 2).reshape(data.shape[1], -1).T
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten )
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
    def __init__(self, exclude=None, report=True, eeg=True, meg=False, stim=False, eog=False):
        self.exclude = exclude
        self.repo_data = {}
        self.repo_images = {}
        self.report = report 
        self.eeg=eeg
        self.meg=meg
        self.stim=stim
        self.eog=eog

    def fit(self, raw):
        dropped = set([ch for ch in raw.ch_names if ch in self.exclude])
        raw = raw.drop_channels(dropped)
        picks_eeg = mne.pick_types(raw.info, eeg=self.eeg, meg=self.meg, stim=self.stim, eog=self.eog)
        raw = raw.pick(picks_eeg)
        return raw


class DetrendEpochs(Transform):
    def __init__(self, detrend_type, report=True):
        self.detrend_type=detrend_type 
        self.report = report
        self.repo_data = {}
        self.repo_images = {}

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
    def __init__(self, baseline, report=True):
        self.baseline=baseline
        self.report = report
        self.repo_data = {}
        self.repo_images = {}

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
    def __init__(self, roi_channels=None, apply=True, report=True):
        self.roi_channels = roi_channels
        self.report = report
        self.repo_data = {}
        self.repo_images = {}
        self.apply = apply

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
    def __init__(self, scenarious_name, report=True):
        self.scenarious_name = scenarious_name
        self.report = report

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
    def __init__(self, stimulus, report=True): 
        self.report = report
        self.stimulus = stimulus
        self.repo_images = {}
        self.repo_data = {}

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


class FilterBandpass(Transform):
    def __init__(self, l_freq, h_freq, notch_freq=None, 
                 method='fir', fir_design='firwin', report=True):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.method = method
        self.fir_design = fir_design
        self.report = report
        self.repo_images = {}
        self.repo_data = {}

    def fit(self, raw):
        raw_filtered = raw.copy()
        if self.notch_freq:
            raw_filtered = raw_filtered.notch_filter(freqs=self.notch_freq, method=self.method, fir_design=self.fir_design, 
                                pad='reflect_limited', phase='zero-double', verbose=False, n_jobs='cuda' if mne.cuda.get_config()['MNE_USE_CUDA']=='true' else -1)
        raw_filtered = raw_filtered.filter(l_freq=self.l_freq, h_freq=self.h_freq, method=self.method, fir_design=self.fir_design, 
                        pad='reflect_limited', phase='zero-double', verbose=False, n_jobs='cuda' if mne.cuda.get_config()['MNE_USE_CUDA']=='true' else -1)
        if self.report:
            fig = compared_spectrum(raw, raw_filtered, fmin=0, fmax=min(100, raw.info['sfreq']/2))
            self.repo_images = {'Spectrum': fig}
            self.repo_data = {}
        return raw_filtered

class StatisticFilter(Transform):
    def __init__(self, method, report=True):
        self.method = method
        self.report = report

    def fit(self, raw, method):
        raw_filtered = raw.copy()
        match method:    
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
                raise f"Unknown method: {method}"
            
        if self.report:
            fig = compared_spectrum(raw, raw_filtered, fmin=0, fmax=100)
            self.repo_images = {'Spectrum': fig}
            self.repo_data = {}
        return raw_filtered
    

class Interpolate(Transform):
    def __init__(self, reset_bads=True, report=True):
        self.reset_bads = reset_bads
        self.report = report
        self.repo_data = {}
        self.repo_images = {}
        
    def fit(self, raw):
        cleaned_raw = raw.copy().interpolate_bads(reset_bads=self.reset_bads, method=dict(eeg="spline"), verbose=False)
        if self.report:
            fig = compared_spectrum(raw, cleaned_raw, fmin=0, fmax=100)
            self.repo_images = {'Spectrum': fig}
            self.repo_data = {}
        return cleaned_raw


class BadChannelsDetector(Transform):
    def __init__(self, method_noise='auto', method_bridge='ed', report=True, mark=True):
        self.method_noise = method_noise
        self.method_bridge = method_bridge
        self.bad_channels = []
        self.electrodesD = {}
        self.report = report
        self.mark = mark

    def fit(self, raw):
        self.electrodesD, clusters, bridge_figs, noised_fig = detect_bad_channels(raw.copy(), self.method_noise, self.method_bridge)
        bad_channels = [ch for sublist in self.electrodesD.values() for ch in sublist]
        united_bad_channels = list(set(bad_channels) | set(raw.info['bads']))
        if self.mark:
            raw.info['bads'] = united_bad_channels

        if self.report:
            self.repo_images = {'Bridged_channels': bridge_figs[0], 'Bridged_hist': bridge_figs[1], 
                                'Noised_channels': noised_fig}
            self.repo_data = {**{'Clusters': clusters}, **self.electrodesD, **{'FINAL': united_bad_channels}}
        return raw


class AutoICA(Transform):
    def __init__(self, n_components='auto', method='picard', max_deleted='auto', 
                brain_threshold=0, exclude_bads=False, report=True):
        self.n_components = n_components
        self.method = method
        self.report = report
        self.repo_data = {}
        self.repo_images = {}
        self.max_deleted = max_deleted
        self.brain_threshold = brain_threshold
        self.exclude_bads = exclude_bads

    def fit(self, raw):
        if self.n_components == 'auto':
            rank = len(raw.ch_names)
            self.n_components=rank
        
        if self.n_components == 'auto' and self.exclude_bads:
            rank = len(raw.ch_names) - len(raw.info['bads']) - 1
            self.n_components=rank

        if self.exclude_bads:
            picks = mne.pick_channels(ch_names=raw.info['ch_names'], include=[], exclude=raw.info["bads"])
        else:
            picks = mne.pick_channels(ch_names=raw.info['ch_names'], include=[])

        ica = ICA(n_components=self.n_components, fit_params=dict(ortho=False, extended=True),
                method=self.method, random_state=97, verbose=False)
        
        ica.fit(raw, picks=picks, verbose=False)

        mne.set_log_level('ERROR') 
        ica_labels = label_components(raw, ica, method='iclabel')

        mne.set_log_level('INFO')
        labels, probas = ica_labels["labels"], ica_labels['y_pred_proba']
        
        ica_sources = ica.get_sources(raw.copy())
        self.sources_data = ica_sources.get_data()
        self.labels = labels
        self.probas = probas 

        cond = lambda label, proba: label not in ['brain'] or (label in ['brain'] and proba <= self.brain_threshold)

        if self.max_deleted == 'auto':
            MAX_DELETED_COMP = ica.n_components_//2
        elif isinstance(self.max_deleted, int):
            MAX_DELETED_COMP = self.max_deleted
        elif self.max_deleted is None:
            MAX_DELETED_COMP = None
            
        exclude_idx    = [idx   for idx, (label, proba) in enumerate(zip(labels, probas)) if cond(label, proba)][:MAX_DELETED_COMP]
        exclude_labels = [label for idx, (label, proba) in enumerate(zip(labels, probas)) if cond(label, proba)][:MAX_DELETED_COMP]

        raw_filtered = ica.apply(raw.copy(), exclude=exclude_idx, verbose=False)
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
            self.repo_data = {'exclude_idx': exclude_idx, 'exclude_labels': exclude_labels}

        return raw_filtered


class Rereference(Transform):   
    def __init__(self, method='average', exclude='bads', report=True):
        self.method = method
        self.exclude = exclude
        self.report = report
        self.repo_data = {}
        self.repo_images = {}
        
    def fit(self, raw):
        raw_filtered = raw.copy()
        picks = mne.pick_channels(ch_names=raw.info['ch_names'], include=[], exclude=raw.info["bads"])
        if self.method=='average' and self.exclude=='bads':
            include_channels = [ch for ch in raw.ch_names if ch not in raw.info['bads']]
            raw_filtered.set_eeg_reference(ref_channels=include_channels, verbose=False)
        else:
            raw_filtered.set_eeg_reference(self.method, verbose=False)
        if self.method == 'laplas':
            raw_filtered = mne.preprocessing.compute_current_source_density(raw_filtered, verbose=False)

        if self.report:
            figs = compared_spectrum(raw.copy().pick(picks), raw_filtered.copy().pick(picks), fmin=0, fmax=100)
            self.repo_data = {}
            self.repo_images = {'Spectrum': figs}

        return raw_filtered


class Resample(Transform):
    def __init__(self, sfreq, report=True):
        self.sfreq = sfreq
        self.report = report
        self.repo_data = {}
        self.repo_images = {}

    def fit(self, raw):
        resampled_raw = raw.copy().resample(sfreq=self.sfreq, npad="auto")
        if self.report:
            self.repo_data = {}
            self.repo_images = {}
        return resampled_raw
    

class BridgeInterpolate(Transform):
    def __init__(self, bridged_idx=None, method_bridge='ed', reset_bridges=True, report=True):
        self.bridged_idx = bridged_idx
        self.method_bridge = method_bridge
        self.report = report
        self.reset_bridges = reset_bridges
        self.repo_data = {}
        self.repo_images = {}

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
        raw = mne.preprocessing.interpolate_bridged_electrodes(raw, self.bridged_idx, bad_limit=len(ch_names))

        if self.reset_bridges:
            raw.info['bads'] = [ch for ch in raw.info['bads'] if ch not in bridged_electrodes]
        return raw


class Raw2Epoch(Transform):
    def __init__(self, tmin=-0.15, tmax=0.75, baseline=(None, 0), stimulus_list=None, scenarious_name=None, report=True):
        self.tmin=tmin 
        self.tmax=tmax 
        self.baseline=baseline
        self.stimulus_list=stimulus_list
        self.scenarious_name = scenarious_name
        self.report = report
        self.repo_data = {}
        self.repo_images = {}

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
    def __init__(self, montage='waveguard64', elc_file=None, mode='Cz', vis=False, threshold=0.08, interpolate=True):
        self.montage = montage
        self.elc_file = elc_file
        self.vis = vis
        self.mode = mode
        self.threshold = threshold
        self.interpolate = interpolate
        self.repo_data = {}
        self.repo_images = {}

    def fit(self, raw):
        return set_montage(raw, self.montage, self.elc_file, self.mode, self.threshold, self.interpolate, self.vis)

