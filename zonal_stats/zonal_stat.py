from pathlib import Path

import fiona
import pandas as pd
import rasterio
from rasterstats import zonal_stats


def mean_and_count_zonal_stat(shape, raster, nodata, list_bands, date_string_start=17, date_string_end=30):
    """Computes zonal stats from all bands using all polygons in a shape file
    Args:
        shape (Path): path to multipolygons shapefile
        raster (Path): path to the ratser file
        nodata: data to ignore when compute stats
        list_bands (list): list of bands to do stats on
        date_string_start (int): position for datetime start in raster fine name
        date_string_end (int): position for datetime end in raster fine name
    Returns:
        stats (dict)
    """

    # get raster band
    with rasterio.open(raster) as src:
        nb_bands = src.count

    # read the shape
    with fiona.open(shape, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        properties = [feature["properties"] for feature in shapefile]

    all_polygon_stats = []
    # loop for each polygon
    for shp, prp in zip(shapes, properties):
        dict_data = {}
        for band in list_bands:
            stats = zonal_stats(shp, raster, stats='mean', band=band, nodata=nodata)
            dict_data.update({f"band_{band}": stats[0]['mean']})

        # update with more information
        dict_data.update(prp)
        dict_data.update({'image_date_time_utc': pd.to_datetime(raster.stem[date_string_start:date_string_end],
                                                                format='%Y%m%dT%H:%M')})
        dict_data.update({'sensor': raster.stem[:3]})

        all_polygon_stats.append(dict_data)

        # add count pixel
        stats = zonal_stats(shp, raster, stats='count', band=nb_bands, nodata=nodata)
        dict_data.update({f"count": stats[0]['count']})
    return all_polygon_stats
