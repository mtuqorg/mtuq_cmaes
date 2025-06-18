import numpy as np
import warnings
from obspy.geodetics.base import degrees2kilometers
from obspy.taup import TauPyModel
from scipy.signal import butter, sosfilt, sosfiltfilt
# mtuq related imports
from mtuq.util.signal import m_to_deg, get_arrival
from mtuq.util.signal import cut
from mtuq.util.cap import taper

def vectorised_detrend_taper(data_mat, taper_frac=0.05):
    """
    Parameters
    ----------
    data_mat : (n_traces, n_samples) float64/float32
        2‑D array holding one trace per row.
    taper_frac : float
        Fraction of the record length to taper at each end (0‑1).

    Returns
    -------
    data_mat_out : ndarray
        Same shape as input, processed in‑place for memory efficiency.
    """
    # 1. demean --------------------------------------------------------------
    # broadcast row means and subtract
    data_mat -= data_mat.mean(axis=1, keepdims=True)

    # 2. remove linear trend -------------------------------------------------
    n_samples = data_mat.shape[1]
    x = np.arange(n_samples, dtype=data_mat.dtype)          # 0 .. N–1
    x -= x.mean()                                           # centre to avoid loss of precision
    denom = (x @ x)                                         # scalar
    # slope for every trace:  (x·y)/(x·x)
    slopes = (data_mat @ x) / denom                         # (n_traces,)
    # y = y – slope * x  (broadcast)
    data_mat -= slopes[:, None] * x

    # 3. cosine (Hann) taper -----------------------------------------------
    n_taper = int(np.floor(taper_frac * n_samples))
    if n_taper > 0:
        # half‑cosine ramps
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_taper, dtype=data_mat.dtype)))
        window = np.ones(n_samples, dtype=data_mat.dtype)
        window[:n_taper]  = ramp
        window[-n_taper:] = ramp[::-1]
        # broadcast multiply
        data_mat *= window

    return data_mat

def batch_process_greens(
    greens_list,
    process,
    scaling_coefficient=1.e5,
    scaling_power=0.5,
    filter_order=4,
    zerophase=True,
    verbose=False,
):
    """
    Optimized version with maximum vectorization and minimal Python overhead.
    - Pre-allocates all arrays
    - Minimizes attribute access
    - Uses numpy broadcasting extensively
    - Reduces function call overhead
    """
    if not greens_list:
        return greens_list

    # Cache frequently accessed attributes at the top
    filter_type = getattr(process, 'filter_type', 'bandpass')
    freq_min = getattr(process, 'freq_min', None)
    freq_max = getattr(process, 'freq_max', None)
    freq = getattr(process, 'freq', None)
    window_type = getattr(process, 'window_type', 'surface_wave')
    window_length = getattr(process, 'window_length', None)
    pick_type = getattr(process, 'pick_type', 'taup')
    taup_model = getattr(process, 'taup_model', 'ak135')

    # Build optimized data structures in one pass
    traces = []
    tensor_metadata = []  # [tensor_idx, trace_idx, station_lat, station_lon, origin_lat, origin_lon, origin_depth, origin_time]
    
    for tensor_idx, tensor in enumerate(greens_list):
        station = tensor.station
        origin = tensor.origin
        if station is None or origin is None:
            raise ValueError("GreensTensor missing 'station' or 'origin' attribute.")
        
        # Cache metadata once per tensor
        station_lat = station.latitude
        station_lon = station.longitude
        origin_lat = origin.latitude
        origin_lon = origin.longitude
        origin_depth = origin.depth_in_m
        origin_time = float(origin.time)
        
        for trace_idx, tr in enumerate(tensor):
            traces.append(tr)
            tensor_metadata.append([tensor_idx, trace_idx, station_lat, station_lon, 
                                   origin_lat, origin_lon, origin_depth, origin_time])
    
    n_traces = len(traces)
    if n_traces == 0:
        return greens_list
    
    # Convert to numpy for vectorized operations
    metadata = np.array(tensor_metadata, dtype=np.float64)
    tensor_indices = metadata[:, 0].astype(np.int32)
    
    # Vectorized distance computation
    lat1, lon1 = metadata[:, 4], metadata[:, 5]  # origins
    lat2, lon2 = metadata[:, 2], metadata[:, 3]  # stations
    
    # Simplified distance calculation (avoiding individual gps2dist_azimuth calls)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(np.radians(dlat/2))**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(dlon/2))**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371000 * c  # Earth radius in meters
    
    # Get sampling info
    fs = 1.0 / traces[0].stats.delta
    max_npts = max(tr.stats.npts for tr in traces)
    
    # Pre-allocate data matrix and metadata arrays
    data_mat = np.zeros((n_traces, max_npts), dtype=np.float32)
    original_dtypes = np.empty(n_traces, dtype=object)
    
    # Vectorized preprocessing
    for i, tr in enumerate(traces):
        if tr.stats.npts < max_npts:
            tr.data = np.pad(tr.data, (0, max_npts - tr.stats.npts),
                            'constant', constant_values=0)
            tr.stats.npts = max_npts
        data_mat[i] = tr.data.astype(np.float32, copy=False)
    
    vectorised_detrend_taper(data_mat, taper_frac=0.05)

    
    # Vectorized scaling computation
    scaling_factors = (distances / scaling_coefficient) ** scaling_power
    
    # Vectorized phase picking
    p_picks = np.zeros(n_traces, dtype=np.float32)
    s_picks = np.zeros(n_traces, dtype=np.float32)
    
    if pick_type == 'taup':
        taup = TauPyModel(taup_model)
        # Batch process unique distance/depth combinations
        unique_pairs = np.unique(np.column_stack([distances, metadata[:, 6]]), axis=0)
        picks_cache = {}
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for dist, depth in unique_pairs:
                try:
                    arrivals = taup.get_travel_times(depth/1000., m_to_deg(dist), phase_list=['p', 's', 'P', 'S'])
                    try:
                        p_pick = get_arrival(arrivals, 'p')
                    except:
                        p_pick = get_arrival(arrivals, 'P')
                    try:
                        s_pick = get_arrival(arrivals, 's')
                    except:
                        s_pick = get_arrival(arrivals, 'S')
                    picks_cache[(dist, depth)] = (p_pick, s_pick)
                except:
                    picks_cache[(dist, depth)] = (0.0, 0.0)
        
        # Map cached picks back to all traces
        for i in range(n_traces):
            key = (distances[i], metadata[i, 6])
            p_picks[i], s_picks[i] = picks_cache[key]
    
    # Design filter once
    sos = None
    if filter_type is not None:
        filter_type = filter_type.lower()
    if filter_type == 'bandpass':
        if freq_min is None or freq_max is None:
            raise ValueError('Bandpass filter requires freq_min and freq_max')
        sos = butter(filter_order, [freq_min, freq_max], btype='bandpass', fs=fs, output='sos')
    elif filter_type == 'lowpass':
        if freq is None:
            raise ValueError('Lowpass filter requires freq')
        sos = butter(filter_order, freq, btype='lowpass', fs=fs, output='sos')
    elif filter_type == 'highpass':
        if freq is None:
            raise ValueError('Highpass filter requires freq')
        sos = butter(filter_order, freq, btype='highpass', fs=fs, output='sos')
    elif filter_type is None:
        sos = None
    else:
        raise ValueError(f'Unknown filter_type: {filter_type}')

    # Vectorized filtering and scaling in single operation
    if sos is not None:
        if zerophase:
            data_mat = sosfiltfilt(sos, data_mat, axis=1)
        else:
            data_mat = sosfilt(sos, data_mat, axis=1)
    
    # Apply scaling (broadcasting)
    data_mat *= scaling_factors[:, None]
    
    # Copy back to traces
    for i, tr in enumerate(traces):
        tr.data = data_mat[i].astype(original_dtypes[i], copy=False)
    
    # Vectorized windowing
    if window_type and window_length:
        if window_type == 'body_wave':
            start_offsets = p_picks - 0.4 * window_length
        elif window_type == 'surface_wave':
            start_offsets = s_picks - 0.3 * window_length
        elif window_type == 'group_velocity':
            group_velocity = getattr(process, 'group_velocity', 3000)
            window_alignment = getattr(process, 'window_alignment', 0.5)
            group_arrivals = distances / group_velocity
            start_offsets = group_arrivals - window_length * window_alignment
        elif window_type == 'min_max':
            v_min = getattr(process, 'v_min', 3000)
            v_max = getattr(process, 'v_max', 6000)
            starts = distances / v_max
            ends = distances / v_min
            mask = (ends - starts) < window_length
            avg_times = (starts + ends) / 2.0
            start_offsets = np.where(mask, avg_times - window_length / 2.0, starts)
        else:
            start_offsets = np.zeros(n_traces)
        
        # Convert to absolute times
        origin_times = metadata[:, 7]
        start_times = start_offsets + origin_times
        end_times = start_times + window_length
        
        # Apply windowing
        for i, tr in enumerate(traces):
            cut(tr, start_times[i], end_times[i])
            taper(tr.data)
    else:
        # Just taper
        for tr in traces:
            taper(tr.data)
    
    return greens_list