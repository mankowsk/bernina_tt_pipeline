from cam_server.pipeline.data_processing import functions, processor
from collections import deque
import numpy as np
from scipy.signal.windows import hann
from scipy.special import erf
initialized = False

def initialize(params):
    global initialized, buffer
    buffer = deque(maxlen=params["buffer_length"])
    initialized = True

def normstep(step):
    """normalizing a test signal for np.correlate"""
    step = step - np.mean(step)
    step = step / np.sum(step**2)
    return step

def get_reference_function(sigma_px=30, reflen=300, window=None):
    rng = reflen / np.sqrt(2) / sigma_px / 2
    ref = -erf(np.linspace(-rng, rng, reflen)) / 2
    if window:
        if window == "hann":
            ref = hann(len(ref)) * ref * 1.64
    ref = normstep(ref)
    return ref

def find_signal(tt_sig, dpx_poly=50, roi=[None,None]):
    """finding signal ref in d.
    ref is expected to be properly normalized
    return position is corrected to center location of the reference signal (as found in signal d)
    """
    # need to invert both to get correct direction
    dark = np.nanmean(buffer, axis=0)
    ratio = (tt_sig/dark)
    ref = get_reference_function(window=None) 
    x0 = (len(ref) + 1) // 2
    c = np.correlate(ratio[slice(*roi)], ref, "valid")

    if roi[0]:
        x0 += np.min(roi)
    p, mx = get_max(c, dpx_poly=dpx_poly, offset=x0)
    return p, mx, c, ratio, dark

def get_max(c, dpx_poly=None, offset=0):
    """getting maximum from a correlation curve (optionally using polynomial fit)"""
    im = c.argmax()
    mx = c[im]

    if dpx_poly:
        poly = np.polyfit(np.arange(-dpx_poly//2,dpx_poly//2),c[im-dpx_poly//2:im+dpx_poly//2], 5)
        root = np.roots(np.polyder(poly,1))
        im = np.real(root[np.argmin(np.abs(root))])
        
    return im + offset, mx

def process(data, pid, timestanp, params):
    if not initialized:
        initialize(params)
    
    tt_sig = data[params["tt_sig"]][::-1]
    events = data[params["events"]]
    #is_laser_dark = events[[params["is_laser_dark"]]]
    #is_fel_dark = events[[params["event_code_xfel_dark"]]]
    is_delayed = events[params["event_code_laser_delayed"]]
    p = None 
    mx = None
    c = None
    p_calib = None
    d = None
    
    if is_delayed:
        buffer.append(tt_sig)
    else:
        if len(buffer) > params["buffer_length"]-1:
            p, mx, c, ratio, dark = find_signal(tt_sig, roi=params["roi"])
            p_calib = np.polyval(params["calibration"], p)*1e15
    edge_results = {"TT_KB:edge_pos": p, "TT_KB:edge_pos_fs": p_calib, "TT_KB:ampl": mx}

    # To populate Chris panel
    edge_results.update({
        "SAROP21-ATT01:edge_pos": p,
        "SAROP21-ATT01:arrival_time": p_calib,
        "SAROP21-ATT01:xcorr": c,
        "SAROP21-ATT01:xcorr_ampl": mx,
        "SAROP21-ATT01:signal": ratio,
        "SAROP21-ATT01:avg_dark_wf": dark,
        "SAROP21-ATT01:raw_wf": tt_sig,
        "SAROP21-ATT01:raw_wf_savgol": tt_sig,
        })
    return edge_results




