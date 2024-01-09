import argparse
import time
import os

import cv2
import fiona
import fiona.transform
import planetary_computer as pc
import pystac_client
import rasterio
import rasterio.errors
import rasterio.mask
import rasterio.transform
import shapely.geometry
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from tqdm import tqdm

STRIDE = 512
CATALOG = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="data/spatial-reasoning", help="Output directory"
    )
    parser.add_argument(
        "--start_idx", type=int, required=False, help="Start index for windows"
    )
    parser.add_argument(
        "--end_idx", type=int, required=False, help="End index for windows"
    )
    return parser


def get_image_from_window(bounds, src_crs):

    minx, miny, maxx, maxy = bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(*bounds))
    warped_geom = fiona.transform.transform_geom(src_crs, "EPSG:4326", geom)
    search = CATALOG.search(
        collections=["naip"], intersects=warped_geom, datetime="2018-01-01/2018-12-31"
    )
    items = search.item_collection()

    best_year = 0
    best_item = None
    for item in items:
        if item.properties["naip:state"] == "md":
            year = int(item.properties["naip:year"])
            if year > best_year:
                best_year = year
                best_item = item

    if best_item is None:
        return None, None

    dst_crs = best_item.properties["proj:epsg"]
    url = best_item.assets["image"].href
    warped_geom = fiona.transform.transform_geom("EPSG:4326", dst_crs, warped_geom)

    with rasterio.open(url) as src:
        with WarpedVRT(src, crs=src_crs, resampling=Resampling.bilinear) as f:
            out_image, _ = rasterio.mask.mask(f, [geom], crop=True, all_touched=True)
            out_image = out_image.transpose(1, 2, 0)
            out_image = cv2.resize(
                out_image, (512, 512), interpolation=cv2.INTER_LINEAR
            )

    out_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, 512, 512)

    return out_image, out_transform


def main(args):

    if not os.path.exists(args.output_dir):
        print("Creating output directory")
        os.makedirs(args.output_dir)

    # Get all windows
    remaining_windows = []
    remaining_idxs = []
    with fiona.open("data/patches.gpkg") as f:
        src_crs = f.crs
        for row in tqdm(f):
            minx = row["properties"]["minx"]
            miny = row["properties"]["miny"]
            maxx = row["properties"]["maxx"]
            maxy = row["properties"]["maxy"]
            idx = row["properties"]["idx"]
            key = (minx, miny, maxx, maxy)

            image_fn = os.path.join(args.output_dir, f"{idx}_image.tif")
            mask_fn = os.path.join(args.output_dir, f"{idx}_mask.tif")
            if os.path.exists(image_fn) and os.path.exists(mask_fn):
                continue
            else:
                assert not os.path.exists(image_fn)
                assert not os.path.exists(mask_fn)
                if args.start_idx is not None and idx < args.start_idx:
                    continue
                if args.end_idx is not None and idx >= args.end_idx:
                    continue
                remaining_windows.append(key)
                remaining_idxs.append(idx)

    # Download all
    for idx, window in tqdm(list(zip(remaining_idxs, remaining_windows))):
        geom = shapely.geometry.mapping(shapely.geometry.box(*window))
        with rasterio.open(
            "data/md_lc_2018_2022-Edition/md_lc_2018_2022-Edition.tif", "r"
        ) as f:
            mask, _ = rasterio.mask.mask(f, [geom], crop=True)

        assert mask.shape == (1, 512, 512)
        mask = mask.squeeze()

        retried = 0
        while retried < 5:
            try:
                out_image, out_transform = get_image_from_window(
                    window, src_crs
                )
                break
            except (pystac_client.exceptions.APIError, rasterio.errors.RasterioIOError):
                retried += 1
                time.sleep(2**retried)
                print(f"Retrying {idx} {retried} times")

        if out_image is None:
            print(f"Skipping {idx}")
            continue

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "nodata": 255,
            "width": 512,
            "height": 512,
            "count": 1,
            "crs": src_crs,
            "transform": out_transform,
            "compress": "lzw",
            "predictor": 2,
            "blockxsize": 128,
            "blockysize": 128,
            "tiled": True,
            "interleave": "pixel",
        }

        with rasterio.open(os.path.join(args.output_dir, f"{idx}_mask.tif"), "w", **profile) as f:
            f.write(mask, 1)

        profile["count"] = 4
        del profile["nodata"]

        with rasterio.open(os.path.join(args.output_dir, f"{idx}_image.tif"), "w", **profile) as f:
            f.write(out_image.transpose(2, 0, 1))

        idx += 1


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    main(args)
