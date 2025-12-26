#!/usr/bin/env python3

import rosbag
import numpy as np
import cv2
import os
from scipy.interpolate import interp1d
import rasterio
from rasterio.transform import from_origin

# ================= CONFIGURACIÓN =================
BAG_FILE = '/home/uib/bagfiles/cabrera/2025_10_29/16_20_34/sparus2_sidescan_2025-10-29-16-20-34_0.bag'
NAV_TOPIC = '/sparus2/navigator/navigation'
OUTPUT_TIFF = 'sss_mosaic_color.tif'

SONAR_RANGE = 30.0      # m
MOSAIC_RES  = 0.1       # m / pixel
BLIND_ZONE  = 0.5       # m
SENSOR_OFFSET = 0.2     # m (offset lateral del SSS)

# ================= FILTRADO =================
def enhance_data(img_input):
    if img_input is None or img_input.size == 0:
        return np.zeros_like(img_input, dtype=np.uint8)

    valid = img_input > 0
    if not np.any(valid):
        return np.zeros_like(img_input, dtype=np.uint8)

    vmin, vmax = np.percentile(img_input[valid], (2, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-5

    img = np.clip((img_input - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)

    img = cv2.medianBlur(img, 5)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    return img

# ================= NAVEGACIÓN (NavSts) =================
def get_nav_data(bag):
    print("-> Leyendo navegación (cola2_msgs/NavSts)...")

    ts, north, east, yaw, alt = [], [], [], [], []

    for _, msg, _ in bag.read_messages(topics=[NAV_TOPIC]):
        if not hasattr(msg, 'header'):
            continue

        ts.append(msg.header.stamp.to_sec())
        north.append(msg.position.north)
        east.append(msg.position.east)
        yaw.append(msg.orientation.yaw)
        alt.append(msg.altitude)

    if len(ts) == 0:
        return None

    ts = np.array(ts)
    idx = np.argsort(ts)

    ts = ts[idx]
    north = np.array(north)[idx]
    east  = np.array(east)[idx]
    yaw   = np.unwrap(np.array(yaw)[idx])
    alt   = np.array(alt)[idx]

    f_n = interp1d(ts, north, bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts, east,  bounds_error=False, fill_value=np.nan)
    f_y = interp1d(ts, yaw,   bounds_error=False, fill_value=np.nan)
    f_h = interp1d(ts, alt,   bounds_error=False, fill_value=np.nan)

    bounds = (np.min(east), np.max(east),
              np.min(north), np.max(north))

    return (f_n, f_e, f_y, f_h), (ts[0], ts[-1]), bounds

# ================= MOSAICO =================
def process_mosaic(bag, nav, time_range, bounds):
    f_n, f_e, f_y, f_h = nav
    t0, t1 = time_range
    min_e, max_e, min_n, max_n = bounds

    margin = SONAR_RANGE + 10
    x_min, x_max = min_e - margin, max_e + margin
    y_min, y_max = min_n - margin, max_n + margin

    width  = int(np.ceil((x_max - x_min) / MOSAIC_RES))
    height = int(np.ceil((y_max - y_min) / MOSAIC_RES))

    print(f"-> Grid: {width} x {height} px")

    grid_p = np.zeros(width * height, dtype=np.float32)
    cnt_p  = np.zeros(width * height, dtype=np.float32)
    grid_s = np.zeros(width * height, dtype=np.float32)
    cnt_s  = np.zeros(width * height, dtype=np.float32)

    def to_idx(e, n):
        c = ((e - x_min) / MOSAIC_RES).astype(np.int32)
        r = ((y_max - n) / MOSAIC_RES).astype(np.int32)
        return c, r

    info = bag.get_type_and_topic_info()
    sss_topics = [t for t in info.topics if "sidescan" in t]

    print("-> Proyectando píxeles SSS...")
    for topic, msg, _ in bag.read_messages(topics=sss_topics):

        if not hasattr(msg, 'header') or not hasattr(msg, 'data'):
            continue

        ts = msg.header.stamp.to_sec()
        if ts < t0 or ts > t1:
            continue

        n = f_n(ts)
        e = f_e(ts)
        yaw = f_y(ts)
        h = f_h(ts)

        if np.isnan(n) or h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8)
        if "port" in topic:
            scan = scan[::-1]
            
        if scan.size < 50:
            continue

        npx = scan.size
        meters_px = SONAR_RANGE / npx
        slant = np.arange(npx) * meters_px
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0))

        valid = ground > BLIND_ZONE
        if not np.any(valid):
            continue

        vals = scan[valid]
        ranges = ground[valid]

        v_right_n = -np.sin(yaw)
        v_right_e =  np.cos(yaw)

        if "port" in topic:
            beam_n = -v_right_n
            beam_e = -v_right_e
            grid, cnt = grid_p, cnt_p
        else:
            beam_n = +v_right_n
            beam_e = +v_right_e
            grid, cnt = grid_s, cnt_s

        # Offset del sensor (una sola vez)
        start_n = n + beam_n * SENSOR_OFFSET
        start_e = e + beam_e * SENSOR_OFFSET

        # Proyección correcta del haz
        px_n = start_n + beam_n * ranges
        px_e = start_e + beam_e * ranges

        c, r = to_idx(px_e, px_n)
        mask = (c >= 0) & (c < width) & (r >= 0) & (r < height)
        idx = r[mask] * width + c[mask]

        np.add.at(grid, idx, vals[mask])
        np.add.at(cnt, idx, 1)

    return grid_p, cnt_p, grid_s, cnt_s, width, height, x_min, y_max

# ================= GUARDADO =================
def save_geotiff(gp, cp, gs, cs, w, h, x_min, y_max):
    print("-> Generando GeoTIFF...")

    port = np.zeros(w * h, dtype=np.float32)
    stbd = np.zeros(w * h, dtype=np.float32)

    mask = cp > 0
    port[mask] = gp[mask] / cp[mask]

    mask = cs > 0
    stbd[mask] = gs[mask] / cs[mask]

    port = enhance_data(port.reshape((h, w)))
    stbd = enhance_data(stbd.reshape((h, w)))

    transform = from_origin(x_min, y_max, MOSAIC_RES, MOSAIC_RES)

    with rasterio.open(
        OUTPUT_TIFF, 'w',
        driver='GTiff',
        height=h, width=w,
        count=3,
        dtype=np.uint8,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(port, 1)   # Rojo: Babor
        dst.write(stbd, 2)   # Verde: Estribor
        dst.write(np.zeros((h, w), dtype=np.uint8), 3)

    print(f"✔ GeoTIFF guardado: {os.path.abspath(OUTPUT_TIFF)}")

# ================= MAIN =================
def main():
    if not os.path.exists(BAG_FILE):
        print("ERROR: Bag no encontrado")
        return

    bag = rosbag.Bag(BAG_FILE)

    nav = get_nav_data(bag)
    if nav is None:
        print("ERROR: navegación vacía")
        return

    nav_funcs, t_range, bounds = nav
    gp, cp, gs, cs, w, h, xmin, ymax = process_mosaic(
        bag, nav_funcs, t_range, bounds
    )

    save_geotiff(gp, cp, gs, cs, w, h, xmin, ymax)
    bag.close()

if __name__ == "__main__":
    main()
