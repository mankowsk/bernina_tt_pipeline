from collections import deque
import numpy as np
from scipy.signal.windows import hann
from scipy.special import erf
initialized = False

def initialize(params):
    global initialized, buffer
    buffer = deque(maxlen=params["dark_buffer_length"])
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

def find_signal(tt_sig, ref, roi=[None, None], dpx_poly = None):
    """finding signal ref in d.
    ref is expected to be properly normalized
    return position is corrected to center location of the reference signal (as found in signal d)
    """
    # need to invert both to get correct direction

    dark = np.nanmean(buffer, axis=0)
    ratio = (tt_sig/dark)

    x0 = (len(ref) + 1) // 2
    c = np.correlate(ratio[roi[0]:roi[1]], ref, "valid")

    if roi[0]:
        x0 += np.min(roi)
    p, mx = get_max(c, dpx_poly=dpx_poly, offset=x0)
    return p, mx, c, ratio, dark

def get_max(c, dpx_poly=None, offset=0):
    """getting maximum from a correlation curve (optionally using polynomial fit)"""
    im = c.argmax()
    mx = c[im]

    if dpx_poly:
        try:
            poly = np.polyfit(np.arange(-dpx_poly//2,dpx_poly//2),c[im-dpx_poly//2:im+dpx_poly//2], 5)
            root = np.roots(np.polyder(poly,1))
            im+= np.real(root[np.argmin(np.abs(root))])
        except:
            im=im
    return im + offset, mx

def process(data, pid, timestanp, params):
    if not initialized:
        initialize(params)
    
    ## initialize all values sent to BS stream to None
    edge_pos = None
    arrival_time = None
    xcorr = None
    xcorr_ampl = None
    ratio = None
    dark = None
    tt_sig = None

    
    ## params used for evaluation    
    roi = params["roi"]
    dpx_poly = params["dpx_poly"]
    sigma_px = params["sigma_px"]
    reflen = params["reflen"]
    window = params["window"]
    
    ## get data and events for sorting
    tt_sig = data[params["tt_sig"]][::-1]
    events = data[params["events"]]
    is_laser_dark = params["is_laser_dark"]
    is_fel_dark = params["is_fel_dark"]
    if is_fel_dark:
        is_fel_dark = events[params["event_code_fel"]]
    is_delayed = events[params["event_code_laser"]]

    # analyze
    if (not is_delayed) & (not is_fel_dark):
        if len(buffer) > params["dark_buffer_length"]-1:
            ref = get_reference_function(sigma_px=sigma_px, reflen=reflen, window=window) 
            edge_pos, xcorr_ampl, xcorr, ratio, dark = find_signal(tt_sig, ref, roi=roi, dpx_poly=dpx_poly)
            arrival_time = np.polyval(params["calibration"], edge_pos)*1e15

    # or add to dark reference
    elif not (is_delayed) & (is_laser_dark):
        buffer.append(tt_sig)

    # To populate Chris panel
    edge_results = {
        "SAROP21-ATT01:edge_pos": edge_pos,
        "SAROP21-ATT01:arrival_time": arrival_time,
        "SAROP21-ATT01:xcorr": xcorr,
        "SAROP21-ATT01:xcorr_ampl": xcorr_ampl,
        "SAROP21-ATT01:signal": ratio,
        "SAROP21-ATT01:avg_dark_wf": dark,
        "SAROP21-ATT01:raw_wf": tt_sig,
        "SAROP21-ATT01:raw_wf_savgol": tt_sig,
        }
    return edge_results


