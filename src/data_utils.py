import ee
from geemap import ee_to_geopandas
import pandas as pd
import boto3
from sklearn.utils import shuffle


def points_to_df(pts_geo):
    """Converts a feature collection into a pandas dataframe

    Args:
        pts_geo: collection of pixels on an image

    Returns:
        df_geo: dataframe containing bands and coordinates of pixels
    """
    df_geo = ee_to_geopandas(pts_geo)
    df_geo = df_geo.drop_duplicates()
    df_geo["x"] = df_geo["geometry"].x
    df_geo["y"] = df_geo["geometry"].y
    df_geo = df_geo.drop("geometry", axis=1)

    return df_geo


def satellite_data(collection, region_pt, date_range):
    """Returns image data from a landsat collection.

    Args:
        collection: dataset name
        region_pt: coordinates of location the image must contain
        date_range: first and last dates to use

    Returns:
        object: a single satellite image
    """
    return (
        ee.ImageCollection(collection)
        .filterBounds(ee.Geometry.Point(region_pt))
        .filterDate(date_range[0], date_range[1])
        .sort("CLOUD_COVER")
        .first()
    )


def sample_points(image, region, scale, num_pix, geom=True, seed=1234):
    """Sample points from dataset

    Args:
        image: image to sample from
        region: region to sample from
        scale: pixel size in meters
        num_pix: number of pixels to be sampled
        geom: whether to add the center of the sampled pixel as property
        seed: random seed used for sampling

    Returns:
        object: ee.FeatureCollection
    """
    return image.sample(
        **{
            "region": region,
            "scale": scale,
            "numPixels": num_pix,
            "seed": seed,
            "geometries": geom,
        }
    )


def get_masks(base_image):
    """Returns image masks corresponding to mangrove and non-mangrove pixels

    Args:
        base_image: earth engine image to create masks from

    Returns:
        objects: ee.Image, ee.Image
    """
    img_mangrove = get_mangrove_data()
    mangrove_mask = base_image.updateMask(img_mangrove.eq(1))
    non_mangrove_mask = base_image.updateMask(mangrove_mask.unmask().Not())

    return mangrove_mask, non_mangrove_mask


def get_data_by_zone_year(
    area_of_int,
    date_range,
    base_dataset,
    bands,
    scale=30,
    num_pix={"minor": 10000, "major": 1000},
):
    """Returns sampled data points from an area of interest

    Args:
        area_of_int: tuple containing (longitude, latitude) of the point of interest
        date_range: list of two strings of format yyyy-mm-dd
        base_dataset: name of satellite data to sample points from
        bands: satellite image bands to keep in dataset
        scale: pixel size in meters for sampling points
        num_pix: dictionary containing number of pixels to sample for two classes

    Returns:
        object: dict
    """
    base_image = satellite_data(base_dataset, area_of_int, date_range)
    base_image = base_image.select(bands)

    mangrove_mask, non_mangrove_mask = get_masks(base_image)

    # sample points from mangrove area
    pts_mangrove = sample_points(
        mangrove_mask, mangrove_mask.geometry(), scale, num_pix["minor"]
    )
    mangrove_gdf = points_to_df(pts_mangrove)
    mangrove_gdf["label"] = 1

    # sample points from non-mangrove area
    pts_non_mangrove = sample_points(
        non_mangrove_mask, non_mangrove_mask.geometry(), scale, num_pix["major"]
    )
    non_mangrove_gdf = points_to_df(pts_non_mangrove)
    non_mangrove_gdf["label"] = 0

    return {
        "base_image": base_image,
        "mangrove_points": pts_mangrove,
        "other_points": pts_non_mangrove,
        "df_mangrove": mangrove_gdf,
        "df_other": non_mangrove_gdf,
    }


def save_regional_data(data_dict, meta_dict, bucket):
    """Uploads the labeled data for a region to s3

    Args:
        data_dict: dictionary containing base image, mangrove and non-mangrove data frames
        meta_dict: dictionary containing metadata
        bucket: s3 bucket name
    """
    df_training = pd.concat([data_dict["df_mangrove"], data_dict["df_other"]], axis=0)
    df_training = shuffle(df_training)

#     fname = f"{meta_dict['src_dataset']}_year{meta_dict['year']}_{meta_dict['poi']}.csv"
    fname = f"{meta_dict['src_dataset']}/Year{meta_dict['year']}/{meta_dict['poi']}.csv"
    df_training.to_csv(f"s3://{bucket}/{fname}", index=False)

    num_rows = df_training.label.value_counts()
    print(
        f"rows: {len(df_training)}, rows_mangrove = {num_rows[1]},  rows_other = {num_rows[0]}"
    )


def split_dataset(test_set_names, bucket, folder):
    """Splits S3 dataset into training and test by region

    Args:
        test_set_names: list of region names for test dataset
        folder: folder name within S3 bucket
        bucket: S3 bucket name
    """
    s3_client = boto3.client("s3")
    items = s3_client.list_objects_v2(Bucket=bucket, Prefix=folder)

    list_train = []
    list_test = []
    for item in items["Contents"]:
        file = item["Key"].split("/")[-1]
        if file.endswith(".csv"):
            list_train.append(file)

    for file_name in list_train:
        for pattern in test_set_names:
            if pattern in file_name:
                list_test.append(file_name)
                list_train.remove(file_name)
                continue

    df_train = pd.concat(
        [pd.read_csv(f"s3://{bucket}/{folder}/{item}") for item in list_train], axis=0
    )
    df_test = pd.concat(
        [pd.read_csv(f"s3://{bucket}/{folder}/{item}") for item in list_test], axis=0
    )

    # save dataframes with coordinates for plotting
    df_train.to_csv(f"s3://{bucket}/{folder}/train_with_coord.csv", index=False)
    df_test.to_csv(f"s3://{bucket}/{folder}/test_with_coord.csv", index=False)
    
    # remove the coordinates for training
    df_train = df_train.drop(["x", "y"], axis=1)
    df_test = df_test.drop(["x", "y"], axis=1)
    df_train.to_csv(f"s3://{bucket}/{folder}/train.csv", index=False)
    df_test.to_csv(f"s3://{bucket}/{folder}/test.csv", index=False)


def get_mangrove_data():
    """
    Returns an earth engine image of mangroves around the globe
    """
    return ee.ImageCollection("LANDSAT/MANGROVE_FORESTS").first()


def main():
    """
    Dataset preparation for mangrove classifier training
    """
    # select bucket to store dataset
    s3_bucket = "sagemaker-gis"
    
    # select satellite data, year and bands
    base_sat_data = "LANDSAT/LC08/C01/T1_SR"
    year = 2015
    bands = "B[1-7]"

    meta_dict = {"src_dataset": base_sat_data.replace("/", "_"), "year": year}
    date_range = [f"{year}-01-01", f"{year}-12-31"]

    # read representative coordinates for each region
    df_zones = pd.read_csv("zones.csv").set_index("region")

    # create dataset for each region
    for area in df_zones.index:
        print(f"processing data for {area}...")
        point_of_int = df_zones.loc[area, ["lon", "lat"]].tolist()
        data_dict = get_data_by_zone_year(
            point_of_int, date_range, base_sat_data, bands
        )
        meta_dict["poi"] = area.replace(" ", "_")
        save_regional_data(data_dict, meta_dict, s3_bucket)

    # split the dataset between training and test sets
    areas_for_test = ["Vietnam2", "Myanmar3", "Cuba2", "India"]
    folder = f"{meta_dict['src_dataset']}/Year{meta_dict['year']}"
    split_dataset(areas_for_test, s3_bucket, folder)


if __name__ == "__main__":
    ee.Initialize()
    main()
