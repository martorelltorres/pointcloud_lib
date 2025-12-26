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
OUTPUT_TIFF = 'sss_mosaic_final.tif'

SONAR_RANGE = 30.0      # m
MOSAIC_RES  = 0.1       # m / pixel
BLIND_ZONE  = 0.5       # m
SENSOR_OFFSET = 0.2     # m (offset lateral del SSS)

# --- PARÁMETROS DE PROCESADO ---

# 1. Corrección de Beam Pattern (Brillo uniforme en todo el rango)
ENABLE_BP_CORRECTION = False   
BP_SMOOTHING_SIGMA   = 20     # Suavizado de la curva de corrección

# 2. Separación de color (Visualización)
# True:  Babor en ROJO, Estribor en VERDE (útil para depurar solapamientos)
# False: Escala de GRISES (mosaico estándar unificado)
ENABLE_COLOR_SIDE_SPLIT = False

# ================= FILTRADO Y UTILIDADES =================
def enhance_data(img_input):
    """Normaliza, ecualiza y enfoca la imagen para mejorar contraste."""
    if img_input is None or img_input.size == 0:
        return np.zeros_like(img_input, dtype=np.uint8)

    valid = img_input > 0
    if not np.any(valid):
        return np.zeros_like(img_input, dtype=np.uint8)

    # 1. Clip de percentiles (eliminar outliers muy brillantes/oscuros)
    vmin, vmax = np.percentile(img_input[valid], (2, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-5

    img = np.clip((img_input - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)

    # 2. Suavizado de ruido de sal y pimienta
    img = cv2.medianBlur(img, 5)

    # 3. CLAHE (Ecualización de histograma adaptativo local)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4. Sharpening (Enfoque de bordes)
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    return img

def smooth_curve(y, sigma):
    """Suaviza una curva 1D usando un kernel Gaussiano."""
    ksize = int(sigma * 6) | 1
    if ksize < 3: return y
    y_smooth = cv2.GaussianBlur(y.reshape(1, -1), (ksize, 1), sigma)
    return y_smooth.flatten()

# ================= MODELADO DE BEAM PATTERN =================
def train_beam_model(bag):
    """Calcula la curva de ganancia inversa basada en estadísticas del log."""
    print("-> Entrenando modelo de corrección de Beam Pattern...")
    
    info = bag.get_type_and_topic_info()
    sss_topics = [t for t in info.topics if "sidescan" in t]
    
    stats = {
        'port': {'sum': None, 'count': None},
        'stbd': {'sum': None, 'count': None}
    }
    
    for topic, msg, _ in bag.read_messages(topics=sss_topics):
        if not hasattr(msg, 'data'): continue
            
        scan = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)
        side = 'port' if 'port' in topic else 'stbd'
        
        if side == 'port': scan = scan[::-1]
        n = scan.size
        if n == 0: continue

        if stats[side]['sum'] is None or stats[side]['sum'].size != n:
            stats[side]['sum'] = np.zeros(n, dtype=np.float64)
            stats[side]['count'] = np.zeros(n, dtype=np.float64)
        
        stats[side]['sum'] += scan
        stats[side]['count'] += 1

    correction_models = {}
    for side in ['port', 'stbd']:
        s = stats[side]['sum']
        c = stats[side]['count']
        
        if s is None or np.max(c) == 0:
            correction_models[side] = None
            continue
            
        avg_profile = np.divide(s, c, out=np.zeros_like(s), where=c>0)
        avg_profile_smooth = smooth_curve(avg_profile, BP_SMOOTHING_SIGMA)
        
        # Calcular media global (evitando la zona ciega que es 0)
        valid_mask = avg_profile_smooth > 1
        if np.any(valid_mask):
            global_mean = np.mean(avg_profile_smooth[valid_mask])
        else:
            global_mean = 1.0
        
        # Ganancia Inversa
        gain_curve = global_mean / (avg_profile_smooth + 1e-5)
        gain_curve = np.clip(gain_curve, 0.5, 5.0) # Limitar ganancia extrema
        
        correction_models[side] = gain_curve.astype(np.float32)
        
    return correction_models

# ================= NAVEGACIÓN (NavSts) =================
def get_nav_data(bag):
    print("-> Leyendo navegación...")
    ts, north, east, yaw, alt = [], [], [], [], []

    for _, msg, _ in bag.read_messages(topics=[NAV_TOPIC]):
        if not hasattr(msg, 'header'): continue
        ts.append(msg.header.stamp.to_sec())
        north.append(msg.position.north)
        east.append(msg.position.east)
        yaw.append(msg.orientation.yaw)
        alt.append(msg.altitude)

    if len(ts) == 0: return None

    ts = np.array(ts)
    idx = np.argsort(ts)
    ts, north, east, alt = ts[idx], np.array(north)[idx], np.array(east)[idx], np.array(alt)[idx]
    yaw = np.unwrap(np.array(yaw)[idx])

    f_n = interp1d(ts, north, bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts, east,  bounds_error=False, fill_value=np.nan)
    f_y = interp1d(ts, yaw,   bounds_error=False, fill_value=np.nan)
    f_h = interp1d(ts, alt,   bounds_error=False, fill_value=np.nan)

    bounds = (np.min(east), np.max(east), np.min(north), np.max(north))
    return (f_n, f_e, f_y, f_h), (ts[0], ts[-1]), bounds

# ================= MOSAICO =================
def process_mosaic(bag, nav, time_range, bounds, beam_corrections=None):
    f_n, f_e, f_y, f_h = nav
    t0, t1 = time_range
    min_e, max_e, min_n, max_n = bounds

    margin = SONAR_RANGE + 10
    x_min, x_max = min_e - margin, max_e + margin
    y_min, y_max = min_n - margin, max_n + margin

    width  = int(np.ceil((x_max - x_min) / MOSAIC_RES))
    height = int(np.ceil((y_max - y_min) / MOSAIC_RES))

    print(f"-> Grid: {width} x {height} px")

    # Grids separados para babor y estribor
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
        if not hasattr(msg, 'header') or not hasattr(msg, 'data'): continue

        ts = msg.header.stamp.to_sec()
        if ts < t0 or ts > t1: continue

        n, e, yaw, h = f_n(ts), f_e(ts), f_y(ts), f_h(ts)
        if np.isnan(n) or h < 0.2: continue

        scan = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)
        
        side = 'stbd'
        if "port" in topic:
            side = 'port'
            scan = scan[::-1]
            
        if scan.size < 50: continue

        # === APLICAR CORRECCIÓN SI ESTÁ HABILITADA ===
        if beam_corrections and beam_corrections.get(side) is not None:
            gain = beam_corrections[side]
            if gain.size == scan.size:
                scan = scan * gain
            else:
                # Interpolación simple si el tamaño difiere
                scan = scan * np.interp(np.linspace(0,1,scan.size), np.linspace(0,1,gain.size), gain)
        # ============================================

        # Geometría slant-range
        npx = scan.size
        meters_px = SONAR_RANGE / npx
        slant = np.arange(npx) * meters_px
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0))

        valid = ground > BLIND_ZONE
        if not np.any(valid): continue

        vals = scan[valid]
        ranges = ground[valid]

        v_right_n, v_right_e = -np.sin(yaw), np.cos(yaw)

        if side == 'port':
            beam_n, beam_e = -v_right_n, -v_right_e
            grid, cnt = grid_p, cnt_p
        else:
            beam_n, beam_e = +v_right_n, +v_right_e
            grid, cnt = grid_s, cnt_s

        start_n = n + beam_n * SENSOR_OFFSET
        start_e = e + beam_e * SENSOR_OFFSET
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

    # Calcular promedios (Raw Float Data)
    raw_port = np.zeros(w * h, dtype=np.float32)
    raw_stbd = np.zeros(w * h, dtype=np.float32)

    mask_p = cp > 0
    raw_port[mask_p] = gp[mask_p] / cp[mask_p]

    mask_s = cs > 0
    raw_stbd[mask_s] = gs[mask_s] / cs[mask_s]

    transform = from_origin(x_min, y_max, MOSAIC_RES, MOSAIC_RES)

    with rasterio.open(
        OUTPUT_TIFF, 'w',
        driver='GTiff',
        height=h, width=w,
        count=3, # Siempre escribimos 3 canales (RGB)
        dtype=np.uint8,
        transform=transform,
        nodata=0
    ) as dst:

        if ENABLE_COLOR_SIDE_SPLIT:
            print("   Modo: Color Separado (R=Babor, G=Estribor)")
            # Procesar cada canal independientemente
            band_r = enhance_data(raw_port.reshape((h, w)))
            band_g = enhance_data(raw_stbd.reshape((h, w)))
            band_b = np.zeros((h, w), dtype=np.uint8) # Azul vacío
            
            dst.write(band_r, 1)
            dst.write(band_g, 2)
            dst.write(band_b, 3)
            
        else:
            print("   Modo: Escala de Grises (Fusión Babor + Estribor)")
            # Fusionar datos brutos ANTES de mejorar contraste para homogeneidad
            # Usamos maximum para que si hay solapamiento, quede el píxel más fuerte
            raw_combined = np.maximum(raw_port, raw_stbd)
            
            img_gray = enhance_data(raw_combined.reshape((h, w)))
            
            # Escribir lo mismo en R, G y B para gris real
            dst.write(img_gray, 1)
            dst.write(img_gray, 2)
            dst.write(img_gray, 3)

    print(f"✔ GeoTIFF guardado: {os.path.abspath(OUTPUT_TIFF)}")

# ================= MAIN =================
def main():
    if not os.path.exists(BAG_FILE):
        print("ERROR: Bag no encontrado")
        return

    bag = rosbag.Bag(BAG_FILE)

    nav = get_nav_data(bag)
    if nav is None: return
    nav_funcs, t_range, bounds = nav

    # 1. Beam Pattern Check
    beam_corrections = None
    if ENABLE_BP_CORRECTION:
        beam_corrections = train_beam_model(bag)
    else:
        print("-> Corrección de Beam Pattern DESACTIVADA.")

    # 2. Process
    gp, cp, gs, cs, w, h, xmin, ymax = process_mosaic(
        bag, nav_funcs, t_range, bounds, beam_corrections
    )

    # 3. Save (usa la config de color interna)
    save_geotiff(gp, cp, gs, cs, w, h, xmin, ymax)
    bag.close()

if __name__ == "__main__":
    main()