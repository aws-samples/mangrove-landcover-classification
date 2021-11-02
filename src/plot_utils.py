import ee
import matplotlib.pyplot as plt
import geemap.eefolium as geemap


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
        
        
def plot_inference(img_region, df_preds, plot_feats, plot_class, legend_dict=None):
    """"""
    region_map = geemap.Map()
    region_map.add_basemap('SATELLITE')
    
    if legend_dict is None:
        legend_dict = {
            "Mangrove Area": (255, 1, 1),
            "Non-mangrove Area": (128, 128, 128),
            "Incorrect Prediction": (0, 0, 0),
            'Correct Prediction': (1, 1, 255)
        }
    
    mangroves_vis = {
        min: 0,
        max: 1.0,
        'palette': ['d40115'],
    }
    
    # collect the correct and incorrect predictions
    df_correct = df_preds.loc[df_preds["y_true"] == df_preds["y_pred"]].copy()
    df_incorrect = df_preds.loc[df_preds["y_true"] != df_preds["y_pred"]].copy()
    
    # convert the pixels into an ee feature collection
    points_correct = ee.FeatureCollection(df_correct[df_correct.y_pred==plot_class].geometry.tolist())
    points_incorrect = ee.FeatureCollection(df_incorrect[df_incorrect.y_pred==plot_class].geometry.tolist())

    # plot the mangrove image for the region
    if plot_class:
        region_map.addLayer(img_region, mangroves_vis, 'Mangroves: Actual')
        legend_dict.pop("Non-mangrove Area")
    else:
        region_map.addLayer(img_region, {}, 'Other: Actual')
        legend_dict.pop("Mangrove Area")

    # plot the plot_class pixels predicted correctly
    region_map.addLayer(points_correct, 
                          {'color': 'blue'}, 
                          'Correct Prediction', 
                          True)
    
    # plot the plot_class pixels predicted incorrectly
    region_map.addLayer(points_incorrect, 
                          {'color': 'black'}, 
                          "Incorrect Prediction", 
                          True)
    
    # center the map and add legend
    region_map.centerObject(plot_feats["poi"], plot_feats["zoom"])
    region_map.add_legend(legend_dict=legend_dict)

    return region_map
