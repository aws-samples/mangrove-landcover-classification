import matplotlib.pyplot as plt


def plot_bands(df_region, bands=None):
    """Plot distribution of pixel band values.

    Args:
        df_region: Dataframe containing band values and mangrove label
        bands: List of bands to be plotted
    """
    if bands is None:
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    ncols = 4
    nrows = len(bands) // ncols + 1

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(16, 8), sharey=True, sharex=True
    )

    row_index = 0
    col_index = 0
    for band in bands:
        df_region.groupby("label")[band].plot(
            kind="kde", ax=axs[row_index][col_index], title=band
        )
        col_index += 1
        if col_index > ncols - 1:
            row_index += 1
            col_index = 0

    for ax in plt.gcf().axes:
        ax.legend(["other", "mangrove"], loc=1)
