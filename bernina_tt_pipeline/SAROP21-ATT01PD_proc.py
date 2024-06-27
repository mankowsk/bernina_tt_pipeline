from collections import deque
from logging import getLogger

import numpy as np
from scipy.signal import savgol_filter

_logger = getLogger(__name__)

initialized = False


def initialize(params):
    global initialized, buffer_savgol, device, step_length, edge_type, refinement, dark_event, fel_on_event, use_dark, calib, use_filter, filter_window, buffer

    device = params["device"]
    step_length = params["step_length"]
    edge_type = params["edge_type"]
    refinement = params["refinement"]
    dark_event = params["dark_event"]
    fel_on_event = params["fel_on_event"]
    buffer_savgol = deque(maxlen=params["buffer_length"])
    use_dark = params["use_dark"]
    calib = params["calib"]
    filter_window = params["filter_window"]
    # use_filter = params['filter']
    buffer = deque(maxlen=params["buffer_length"])
    initialized = True


def _interpolate_row(y_known, x_known, x_interp):
    y_interp = np.interp(x_interp, x_known, y_known)
    return y_interp


def find_edge(data, step_length=50, edge_type="falling", refinement=1):
    # refine data
    data_length = data.shape[1]
    refined_data = np.apply_along_axis(
        _interpolate_row,
        axis=1,
        arr=data,
        x_known=np.arange(data_length),
        x_interp=np.arange(0, data_length - 1, refinement),
    )

    # prepare a step function and refine it
    step_waveform = np.ones(shape=(step_length,))
    if edge_type == "rising":
        step_waveform[: int(step_length / 2)] = -1
    elif edge_type == "falling":
        step_waveform[int(step_length / 2) :] = -1

    step_waveform = np.interp(
        x=np.arange(0, step_length - 1, refinement), xp=np.arange(step_length), fp=step_waveform
    )

    # find edges
    xcorr = np.apply_along_axis(np.correlate, 1, refined_data, v=step_waveform, mode="valid")
    edge_position = np.argmax(xcorr, axis=1).astype(float) * refinement
    xcorr_amplitude = np.amax(xcorr, axis=1)

    # correct edge_position for step_length
    edge_position += np.floor(step_length / 2)

    return {
        "edge_pos": edge_position,
        "xcorr": xcorr,
        "xcorr_ampl": xcorr_amplitude,
        "signal": data,
    }


def process(data, pulse_id, timestamp, params):
    if not initialized:
        initialize(params)
    output = {}

    # Read stream inputs
    prof_sig = data[params["prof_sig"]]
    try:
        prof_sig_savgol = savgol_filter(prof_sig, filter_window, 3)
    except:
        output[f"{device}:raw_wf"] = prof_sig
        return output  # added for intermitent cases with prof_sig shorter than filter window
    events = data[params["events"]]

    if events[dark_event] and use_dark:
        buffer.append(prof_sig)
    if prof_sig_savgol.ndim == 1:
        prof_sig_savgol = prof_sig_savgol[np.newaxis, :]

    if events[dark_event] and use_dark:
        buffer_savgol.append(prof_sig_savgol)
        edge_results = {"edge_pos": None, "xcorr": None, "xcorr_ampl": None, "signal": None}
    else:
        if events[fel_on_event] and buffer_savgol:
            prof_sig_norm = prof_sig_savgol / np.mean(buffer_savgol, axis=0)
            edge_results = find_edge(prof_sig_norm, step_length, edge_type, refinement)
        elif events[fel_on_event] and not use_dark:
            edge_results = find_edge(prof_sig_savgol, step_length, edge_type, refinement)
        else:
            edge_results = {"edge_pos": None, "xcorr": None, "xcorr_ampl": None, "signal": None}

    # # calib edge
    # edge_results["arrival_time"] = np.polyval(calib, edge_results["edge_pos"])
    # # sort edge by parity
    # if pulse_id % 2 == 0:
    #     try:
    #         edge_results["arrival_time_even"] = edge_results["edge_pos"] * calib
    #     except:
    #         edge_results["arrival_time_even"] = None
    #     edge_results["arrival_time_odd"] = None
    # else:
    #     edge_results["arrival_time_even"] = None
    #     try:
    #         edge_results["arrival_time_odd"] = edge_results["edge_pos"] * calib
    #     except:
    #         edge_results["arrival_time_odd"] = None
    # # push pulse ID for debuging
    # edge_results["pulse_id"] = pulse_id
    # Set bs outputs
    for key, value in edge_results.items():
        output[f"{device}:{key}"] = value

    # output[f"{device}:raw_wf"] = prof_sig
    # output[f"{device}:raw_wf_savgol"] = prof_sig_savgol

    # if events[dark_event]:
    #     output[f"{device}:dark_wf"] = prof_sig
    #     output[f"{device}:dark_wf_savgol"] = prof_sig_savgol
    # else:
    #     output[f"{device}:dark_wf"] = None
    #     output[f"{device}:dark_wf_savgol"] = None

    # if buffer:
    #     output[f"{device}:avg_dark_wf"] = np.mean(buffer, axis=0)
    # else:
    #     output[f"{device}:avg_dark_wf"] = None

    # if buffer_savgol:
    #     output[f"{device}:avg_dark_wf_savgol"] = np.mean(buffer_savgol, axis=0)
    # else:
    #     output[f"{device}:avg_dark_wf_savgol"] = None

    return output

