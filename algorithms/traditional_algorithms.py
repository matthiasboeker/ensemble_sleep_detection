import numpy as np
from hmmlearn import hmm
import pandas as pd


def specific_frequency_filter(time_series, sampling_rate, frequency, bandwidth):
    # Compute the Fourier transform
    fft_vals = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(time_series.size, d=1 / sampling_rate)
    fft_vals[np.abs(frequencies) < (frequency - bandwidth / 2)] = 0
    fft_vals[np.abs(frequencies) > (frequency + bandwidth / 2)] = 0
    filtered_time_series = np.fft.ifft(fft_vals)
    return np.real(filtered_time_series)


def lh_pass_filter(time_series, sampling_rate, lower_bound, upper_bound):
    fft_vals = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(time_series.size, d=1 / sampling_rate)
    fft_vals[np.abs(frequencies) < lower_bound] = 0
    fft_vals[np.abs(frequencies) > upper_bound] = 0
    filtered_time_series = np.fft.ifft(fft_vals)
    return np.real(filtered_time_series)


def reorder_states_by_means(model, hidden_states):
    means = model.means_.flatten()
    order = np.argsort(means)
    model.means_ = model.means_[order]
    model.covars_ = model.covars_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order, :][:, order]
    hidden_states = np.array([np.where(order == state)[0][0] for state in hidden_states])
    return hidden_states, model


def fourier_based_hmm(signal, epoch_duration, nr_comp=2, lowpass=0.00001, highpass=0.0002):
    filtered = lh_pass_filter(signal, 1/60, lowpass, highpass)
    model = hmm.GaussianHMM(n_components=nr_comp, covariance_type="full", n_iter=100, random_state=123456)
    signal_reshaped = np.expand_dims(filtered, axis=1)
    model.fit(signal_reshaped)
    hidden_states = model.predict(signal_reshaped)
    hidden_states, model = reorder_states_by_means(model, hidden_states)
    return pd.Series(hidden_states, index=signal.index)


def rule_a(series):
    return series.where(~(series.shift(-4).rolling(5).sum() == 1) | (series == 0), 1)


def rule_b(series):
    return series.where(~(series.shift(-10).rolling(11).sum() == 0) | (series == 1), 0)


def rule_c(series):
    return series.where(~(series.shift(-15).rolling(16).sum() == 0) | (series == 1), 0)


def rule_d(series):
    rolling_sum_before = series.shift(-10).rolling(11).sum()
    rolling_sum_after = series.shift(10).rolling(11).sum()
    return series.where(~((series.rolling(11, center=True).sum() <= 6) & ((rolling_sum_before == 0) | (rolling_sum_after == 0))), 0)


def rule_e(series):
    rolling_sum_before = series.shift(-20).rolling(21).sum()
    rolling_sum_after = series.shift(20).rolling(21).sum()
    return series.where(~((series.rolling(21, center=True).sum() <= 10) & ((rolling_sum_before == 0) | (rolling_sum_after == 0))), 0)


def calculate_total_vector(signals_df: pd.DataFrame) -> pd.Series:
    signals_df = signals_df[["x", "y", "z"]]
    return np.sqrt((signals_df**2).sum(axis=1))


def pad_signal(signal, padding_length, mode):
    if mode == 'start':
        return pd.concat([pd.Series([0] * padding_length), signal])
    elif mode == 'end':
        return pd.concat([signal, pd.Series([0] * padding_length)])
    elif mode == 'both':
        return pd.concat([pd.Series([0] * padding_length), signal, pd.Series([0] * padding_length)])
    else:
        raise ValueError("Mode should be 'start', 'end', or 'both'")


def nat_function(x):
    return len([i for i in x if i > 0])


def sadeh_sleep_stage_algo(signal, epoch_length):
    if not isinstance(signal, pd.Series):
        raise TypeError('Input signal should be a pandas Series')
    signal_ = signal.clip(upper=300)
    mean_signal = pad_signal(signal, 5, 'both').rolling(11, center=True).mean()[5:-5].reset_index(drop=True)
    std_signal = pad_signal(signal_, 5, 'start').rolling(6).std()[5:].reset_index(drop=True)
    nat_signal = pad_signal(signal_, 5, 'both').rolling(11, center=True).apply(nat_function, raw=False)[5:-5].reset_index(drop=True)
    log_signal = signal_.apply(lambda x: np.log(x + 1)).reset_index(drop=True)
    sadeh_signal = 7.601 - (0.065 * mean_signal) - (1.08 * nat_signal) - (0.056 * std_signal) - (0.703 * log_signal)
    sleep_stages = (sadeh_signal <= sadeh_signal.describe()["50%"]).astype(int).values
    return pd.Series(sleep_stages, index=signal.index)


def calculate_cole_kripke(signal, epoch_length):
    weights = [0.001*106, 0.001*54, 0.001*58, 0.001*76, 0.001*230, 0.001*74, 0.001*67]
    weights.reverse()
    signal = signal/100
    signal = signal.clip(upper=300)
    weighted_activity = np.convolve(signal.values, weights, mode='full')
    # Truncate the convolved signal to the size of the input signal
    weighted_activity = weighted_activity[:len(signal)]
    sleep_wake = (weighted_activity > 1).astype(int)
    return pd.Series(sleep_wake, index=signal.index)


def sazonova_algorithm(activity_counts, activity_threshold=40, window_size=11, wake_threshold=2):
    """
    Apply the Sazonova sleep-wake algorithm.

    Parameters:
    - activity_counts (pd.Series): The activity counts data.
    - activity_threshold (int): Threshold for an epoch to be considered active.
    - window_size (int): Size of the moving window.
    - wake_threshold (int): Number of active epochs within the window needed to classify the middle epoch as wake.

    Returns:
    pd.Series: Binary series where 1 indicates wake and 0 indicates sleep.
    """

    # Classify epochs as active/inactive
    is_active = activity_counts > activity_threshold

    # Initialize results with 0 (sleep)
    results = pd.Series(0, index=activity_counts.index)

    half_window = window_size // 2

    for i in range(half_window, len(activity_counts) - half_window):
        window_data = is_active.iloc[i - half_window: i + half_window + 1]
        if window_data.sum() > wake_threshold:
            results.iloc[i] = 1  # Wake
    return pd.Series(results, index=activity_counts.index)


def calculate_oakley(activity_counts, threshold, T=40, S=100, N=2):
    n = len(activity_counts)
    results = []

    for i in range(n):
        current_count = activity_counts.iloc[i]

        if current_count > T:
            results.append(1)  # Awake
            continue

        # Define the start and end indices to sum over the surrounding N minutes
        start_index = max(0, i - N)
        end_index = min(n, i + N + 1)  # +1 because Python slices are exclusive at the end

        surrounding_sum = sum(activity_counts[start_index:end_index])

        if surrounding_sum > S:
            results.append(1)  # Awake
        else:
            results.append(0)  # Sleep

    return pd.Series(results, index=activity_counts.index)