#!/usr/bin/env python3

import numpy as np
import rasterio
import cv2
import open3d as o3d
from rasterio.transform import from_origin, Affine
from scipy.interpolate import griddata

# ================= CONFIG =================
MB_MESH = "/home/uib/derelictes_ws/bagfiles/results/mb_surface_mesh.ply"
SSS_TIF = "/home/uib/derelictes_ws/src/pointcloud_lib/scripts/sss_mosaic_final.tif"
OUT_TIF = "sss_registered.tif"

GRID_RES = 0.5          # m
VOXEL_SIZE = 0.5        # m
ICP_DIST = 2.0          # m

CANNY_LOW = 50
CANNY_HIGH = 150
# =========================================


# ------------------------------------------------
def load_sss(path):
    with rasterio.open(path) as src:
        img = src.read(1)
        transform = src.transform
        profile = src.profile
    return img, transform, profile


# ------------------------------------------------
def mesh_to_dem(mesh_path, res):
    print("-> Cargando mesh MB")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pts = np.asarray(mesh.vertices)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xi = np.arange(xmin, xmax, res)
    yi = np.arange(ymax, ymin, -res)
    Xi, Yi = np.meshgrid(xi, yi)

    print("-> Interpolando DEM MB")
    Zi = griddata((x, y), z, (Xi, Yi), method='nearest')
    Zi[np.isnan(Zi)] = np.nanmin(Zi)

    transform = from_origin(xmin, ymax, res, res)
    return Zi, transform


# ------------------------------------------------
def dem_to_hillshade(z, res):
    dx, dy = res, res
    dzdx, dzdy = np.gradient(z, dx, dy)

    slope = np.pi / 2 - np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(-dzdx, dzdy)

    az = np.deg2rad(315)
    alt = np.deg2rad(45)

    hs = (
        np.sin(alt) * np.sin(slope) +
        np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )

    return np.clip(hs, 0, 1)


# ------------------------------------------------
def raster_to_edges(img):
    img_n = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_u8 = img_n.astype(np.uint8)
    return cv2.Canny(img_u8, CANNY_LOW, CANNY_HIGH)


# ------------------------------------------------
def edges_to_points(edges, transform):
    ys, xs = np.where(edges > 0)
    xs_geo, ys_geo = rasterio.transform.xy(transform, ys, xs)
    return np.column_stack((xs_geo, ys_geo, np.zeros(len(xs_geo))))


# ------------------------------------------------
def icp_2d(src_pts, tgt_pts):
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    src = src.voxel_down_sample(VOXEL_SIZE)
    tgt = tgt.voxel_down_sample(VOXEL_SIZE)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt,
        ICP_DIST,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return reg.transformation


# ------------------------------------------------
def apply_transform_to_tif(in_tif, out_tif, T):
    with rasterio.open(in_tif) as src:
        data = src.read()
        tr = src.transform
        profile = src.profile

        dx, dy = T[0, 3], T[1, 3]
        yaw = np.arctan2(T[1, 0], T[0, 0])

        A = Affine.translation(dx, dy) * Affine.rotation(np.degrees(yaw))
        new_tr = tr * A

        profile.update(transform=new_tr)

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(data)

    print(f"✔ SSS registrado guardado en {out_tif}")


# ------------------------------------------------
def main():

    print("-> Cargando SSS")
    sss, sss_tr, _ = load_sss(SSS_TIF)

    print("-> Generando DEM MB")
    dem, mb_tr = mesh_to_dem(MB_MESH, GRID_RES)

    print("-> Hillshade MB")
    mb_img = dem_to_hillshade(dem, GRID_RES)

    print("-> Extrayendo bordes")
    sss_edges = raster_to_edges(sss)
    mb_edges = raster_to_edges(mb_img)

    print("-> Bordes a puntos")
    sss_pts = edges_to_points(sss_edges, sss_tr)
    mb_pts = edges_to_points(mb_edges, mb_tr)

    print("-> ICP 2D")
    T = icp_2d(sss_pts, mb_pts)
    print(T)

    print("-> Aplicando transformación")
    apply_transform_to_tif(SSS_TIF, OUT_TIF, T)


if __name__ == "__main__":
    main()
