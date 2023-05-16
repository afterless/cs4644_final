import torch as t
import numpy as np
import einops
import tqdm.notebook as tqdm

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go


from functools import *
import pandas as pd

import copy
import re

def to_numpy(tensor, flat=False):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return np.array(tensor)
    elif isinstance(tensor, t.Tensor):
        if flat:
            return tensor.flatten().detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()
    elif type(tensor) in [int, float, bool, str]:
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type {type(tensor)}")
    
def melt(tensor):
    arr = to_numpy(tensor)
    n = arr.ndim
    grid = np.ogrid[tuple(map(slice, arr.shape))]
    out = np.empty(arr.shape + (n+1,), dtype=np.result_type(arr.dtype, int))
    offset = 1

    for i in range(n):
        out[..., i+offset] = grid[i]
    out[..., -1+offset] = arr
    out.shape = (-1, n+1)

    df = pd.DataFrame(out, columns=['value']+[str(i) for i in range(n)], dtype=float)

    return df.convert_dtypes([float]+[int]*n) #type: ignore

def broadcast_up(array, shape, axis_str=None):
    n = len(shape)
    m = len(array.shape)
    if axis_str is None:
        axis_str = " ".join([f"x{i}" for i in range(n-m, n)])
    return einops.repeat(array, f"{axis_str}->({' '.join([f'x{i}' for i in range(n)])})", **{f"x{i}": shape[i] for i in range(n)})

# Defining kwargs
DEFAULT_KWARGS = dict(
    xaxis="x",
    yaxis="y",
    range_x=None,
    range_y=None,
    animation_name="snapshot",
    color_name="Color"
    color=None,
    log_x=False,
    log_y=False,
    toggle_x=False,
    toggle_y=False,
    legend=True,
    hover=None,
    hover_name="data"
    return_fig=True,
    animation_index=None,
    line_labels=None,
    markers=False,
    frame_rate=None,
    facet_labels=None,
    debug=False,
    transition="none",
)

def split_kwargs(kwargs):
    custom = dict(DEFAULT_KWARGS)
    plotly = {}
    for k, v in kwargs.items():
        if k in custom:
            custom[k] = v
        else:
            plotly[k] = v
    return custom, plotly

def update_play_button()