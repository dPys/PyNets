"""Plotting functions to embed into html reports."""

import nibabel as nib
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_t1w_with_segs(t1w, wm, csf):
    """Plot t1w images using plotly.

    Parameters
    ----------
    t1w : str
        Path to a t1w image.
    wm : str
        Path to wm image.
    csf : str
        Path to csf image.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A plotly figure that is ready-to-embed using the to_html method.
    """

    # Load data from file
    t1w_arr = nib.load(t1w).get_fdata()
    wm_arr = nib.load(wm).get_fdata()
    csf_arr = nib.load(csf).get_fdata()

    # Init figure
    nrows = 3
    ncols = 7
    fig = make_subplots(nrows, ncols, vertical_spacing=0.005, horizontal_spacing=0.005)

    # Get subplot positions
    fig_idxs = [(row, col) for row in range(1, nrows+1)
                for col in range(1, ncols+1)]

    # Plot segs
    _add_overlay(fig, t1w_arr, "gray", 1.0, fig_idxs)
    _add_overlay(fig, wm_arr, "ice", 0.5, fig_idxs)
    _add_overlay(fig, csf_arr, "ice", 0.5, fig_idxs)

    fig.update_layout(width=800, height=500)

    return fig


def _add_overlay(fig, data_arr, colorscale, opacity, fig_idxs):
    """Add an overlay to the figure."""

    # Space out z_slices
    z_max = np.shape(data_arr)[2]
    pad = 30
    z_slices = np.linspace(pad, z_max-pad, num=21, dtype=int)

    for idx, z_slice in enumerate(z_slices):

        # Get subplot position
        x_pos, y_pos = fig_idxs[idx]

        # Don't plot if array is all zeros
        if np.mean(data_arr[:, :, z_slice]) > 0:

            fig.add_trace(go.Heatmap(z=np.rot90(data_arr[:, :, z_slice], k=3), showscale=False,
                                     colorscale=colorscale, opacity=opacity),
                          x_pos, y_pos)

        fig.update_xaxes(showticklabels=False, row=x_pos, col=y_pos)
        fig.update_yaxes(showticklabels=False, row=x_pos, col=y_pos)
