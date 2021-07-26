import librosa
import numpy as np
import scipy.signal as signal


def compute_mfcc_features(y, sr):
    mfcc_feat = librosa.feature.mfcc(y, sr, n_mfcc=12, n_mels=12, hop_length=int(sr / 100), n_fft=int(sr / 40)).T
    s, phase = librosa.magphase(librosa.stft(y, hop_length=int(sr / 100)))
    rms = librosa.feature.rms(S=s).T
    return np.hstack([mfcc_feat, rms])


def compute_delta_features(mfcc_feat):
    return np.vstack([librosa.feature.delta(mfcc_feat.T), librosa.feature.delta(mfcc_feat.T, order=2)]).T


def collapse_to_start_and_end_frame(instance_list):
    return (instance_list[0], instance_list[-1])


def frame_span_to_time_span(frame_span):
    return (frame_span[0] / 100., frame_span[1] / 100.)


def format_features(mfcc_feat, delta_feat, index, window_size=37):
    return np.append(mfcc_feat[index - window_size:index + window_size],
                     delta_feat[index - window_size:index + window_size])


def lowpass(sig, filter_order=2, cutoff=0.01):
    b, a = signal.butter(filter_order, cutoff, output='ba')
    return signal.filtfilt(b, a, sig)


def get_laughter_instances(probs, threshold=0.5, min_length=0.2):
    instances = []
    current_list = []
    for i in range(len(probs)):
        if np.min(probs[i:i + 1]) > threshold:
            current_list.append(i)
        else:
            if len(current_list) > 0:
                instances.append(current_list)
                current_list = []
    instances = [frame_span_to_time_span(collapse_to_start_and_end_frame(i)) for i in instances if len(i) > min_length]
    return instances


def get_feature_list(y, sr, window_size=37):
    mfcc_feat = compute_mfcc_features(y, sr)
    delta_feat = compute_delta_features(mfcc_feat)
    zero_pad_mfcc = np.zeros((window_size, mfcc_feat.shape[1]))
    zero_pad_delta = np.zeros((window_size, delta_feat.shape[1]))
    padded_mfcc_feat = np.vstack([zero_pad_mfcc, mfcc_feat, zero_pad_mfcc])
    padded_delta_feat = np.vstack([zero_pad_delta, delta_feat, zero_pad_delta])
    feature_list = []
    for i in range(window_size, len(mfcc_feat) + window_size):
        feature_list.append(format_features(padded_mfcc_feat, padded_delta_feat, i, window_size))
    feature_list = np.array(feature_list)
    return feature_list


def segment_laughs(input_path, model, threshold=0.5, min_length=0.2):
    y, sr = librosa.load(input_path, sr=8000)

    feature_list = get_feature_list(y, sr)

    probs = model.predict(feature_list)
    probs = probs.reshape((len(probs),))
    filtered = lowpass(probs)
    instances = get_laughter_instances(filtered, threshold=threshold, min_length=min_length)

    if len(instances) > 0 and instances[0][1] - instances[0][0] >= min_length:
        return ([{'start': i[0], 'end': i[1]} for i in instances])

    else:
        return "No laugh"
