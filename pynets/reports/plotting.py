"""Plotting functions to embed into html reports."""

import nibabel as nib
import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_t1w(t1w):
    """Plot t1w images using plotly.

    Parameters
    ----------
    t1w : str
        Path to a t1w image.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A plotly figure that is ready-to-embed using the to_html method.
    """

    # Load data from file
    t1w_arr = nib.load(t1w).get_fdata()

    # Space out z-slices
    z_max = np.shape(t1w_arr)[2]
    pad = 30
    z_slices = np.linspace(pad, z_max-pad, num=21, dtype=int)

    # Init figure
    nrows = 3
    ncols = 7
    fig = make_subplots(nrows, ncols, vertical_spacing=0.005, horizontal_spacing=0.005)

    # Get subplot positions
    fig_idxs = [(row, col) for row in range(1, nrows+1)
                for col in range(1, ncols+1)]

    for idx, z_slice in enumerate(z_slices):

        # Slice and rotate the t1w array
        img_slice = np.rot90(t1w_arr[:, :, z_slice], k=3)

        # Get subplot coords
        x_coord, y_coord = fig_idxs[idx]

        fig.add_trace(go.Heatmap(z=img_slice, showscale=False, colorscale="gray"),
                      x_coord, y_coord)

        # Update axes
        fig.update_xaxes(showticklabels=False, row=x_coord, col=y_coord)
        fig.update_yaxes(showticklabels=False, row=x_coord, col=y_coord)

    fig.update_layout(width=800, height=500)

    return fig
