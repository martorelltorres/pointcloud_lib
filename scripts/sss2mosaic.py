#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.interpolate import interp1d
import rasterio
from rasterio.transform import from_origin
import pyproj
import os
import sys

# --- CONFIGURACIÓN ---
BAG_FILE = '/home/uib/bagfiles/cabrera/2025_10_27/13_43_54/sparus2_sidescan_2025-10-27-13-43-54_0.bag' 
OUTPUT_TIFF = 'mosaico_ned_final.tif'

# PARÁMETROS DEL SONAR
SONAR_RANGE = 30.0   
BLIND_ZONE = 0.5     
MOSAIC_RES = 0.1     

# FRAMES
FRAME_BASE = "sparus2/base_link"
FRAME_PORT = "sparus2/sidescan_port"
FRAME_STBD = "sparus2/sidescan_starboard"
# ----------------------

class TFManager:
    """Extrae la posición de montaje (Offset) de los sensores desde el bag"""
    def __init__(self):
        self.offsets = {} 

    def load_static_tfs(self, bag):
        print("-> Leyendo TFs estáticas...")
        for topic, msg, t in bag.read_messages(topics=['/tf_static', '/tf']):
            for transform in msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                
                if child in [FRAME_PORT, FRAME_STBD] and parent == FRAME_BASE:
                    tx = transform.transform.translation.x
                    ty = transform.transform.translation.y
                    tz = transform.transform.translation.z
                    self.offsets[child] = np.array([tx, ty, tz])
        
        # Defaults
        if FRAME_PORT not in self.offsets: self.offsets[FRAME_PORT] = np.zeros(3)
        if FRAME_STBD not in self.offsets: self.offsets[FRAME_STBD] = np.zeros(3)
        
        print(f" [INFO] Offset Babor: {self.offsets[FRAME_PORT]}")
        print(f" [INFO] Offset Estribor: {self.offsets[FRAME_STBD]}")

    def get_offset(self, child_frame):
        return self.offsets.get(child_frame, np.zeros(3))

def get_nav_data(bag, topic_nav):
    print(f"\n-> Procesando navegación (Lógica NED pura)...")
    timestamps, northings, eastings, yaws_ned = [], [], [], []
    
    # Proyección: Lat/Lon -> UTM (Metros)
    # Nota: pyproj devuelve (Este, Norte)
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    proj_utm = pyproj.CRS("EPSG:32631") 
    transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True)

    for topic, msg, t in bag.read_messages(topics=[topic_nav]):
        ts = msg.header.stamp.to_sec() if hasattr(msg.header, 'stamp') else t.to_sec()
        try:
            # global_position suele ser Lat/Lon
            east, north = transformer.transform(msg.global_position.longitude, msg.global_position.latitude)
            
            timestamps.append(ts)
            northings.append(north)
            eastings.append(east)
            # Guardamos Yaw NED sin modificar (0=Norte, Clockwise)
            yaws_ned.append(msg.orientation.yaw) 
        except: continue

    if not timestamps: return None

    # Ordenar cronológicamente
    t_arr = np.array(timestamps)
    idx = np.argsort(t_arr)
    t_arr = t_arr[idx]
    
    # Unwrap del yaw
    yaw_continuous = np.unwrap(np.array(yaws_ned)[idx])

    # Interpoladores
    interp_east = interp1d(t_arr, np.array(eastings)[idx], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_north = interp1d(t_arr, np.array(northings)[idx], kind='linear', bounds_error=False, fill_value=np.nan)
    interp_yaw = interp1d(t_arr, yaw_continuous, kind='linear', bounds_error=False, fill_value=np.nan)

    return interp_east, interp_north, interp_yaw, proj_utm, (t_arr[0], t_arr[-1])

def process_sonar_images(bag, nav_data, tf_man):
    print(f"\n-> Proyectando imágenes (Lógica Física Estricta NED)...")
    
    info = bag.get_type_and_topic_info()
    sss_topics = [t for t in info.topics.keys() if "sidescan" in t and "raw_data" in t and "info" not in t]
    
    interp_east, interp_north, interp_yaw, _, (t_start, t_end) = nav_data
    bridge = CvBridge()
    global_x, global_y, intensities = [], [], []
    stats = {'ok': 0}

    for topic, msg, t in bag.read_messages(topics=sss_topics):
        t_img = msg.header.stamp.to_sec() if hasattr(msg.header, 'stamp') else t.to_sec()
        if t_img < t_start or t_img > t_end: continue

        # 1. ESTADO DEL AUV (Yaw NED Original)
        auv_east = interp_east(t_img)
        auv_north = interp_north(t_img)
        yaw_auv = interp_yaw(t_img) 

        if np.isnan(auv_east): continue

        # 2. POSICIÓN REAL DEL SENSOR (Usando TF y trigonometría NED)
        # Calculamos sen/cos del AUV para rotar el offset
        c_auv = np.cos(yaw_auv)
        s_auv = np.sin(yaw_auv)

        if "port" in topic:
            off = tf_man.get_offset(FRAME_PORT)
        else:
            off = tf_man.get_offset(FRAME_STBD)
        
        # Rotar Offset (Body -> NED World)
        # En NED: X (Forward) -> cos(yaw) en Norte, sin(yaw) en Este
        #         Y (Right)   -> -sin(yaw) en Norte, cos(yaw) en Este
        sensor_north = auv_north + (off[0] * c_auv - off[1] * s_auv)
        sensor_east  = auv_east  + (off[0] * s_auv + off[1] * c_auv)

        # ---------------------------------------------------------
        # 3. CÁLCULO EXPLÍCITO DE LA DIRECCIÓN DE DISPARO (CORRECCIÓN)
        # ---------------------------------------------------------
        if "starboard" in topic:
            # Estribor: Dispara a 90 grados en sentido HORARIO (+pi/2)
            beam_yaw = yaw_auv + (np.pi / 2.0)
        else: # Port
            # Babor: Dispara a 90 grados en sentido ANTI-HORARIO (-pi/2)
            beam_yaw = yaw_auv - (np.pi / 2.0)

        # 4. OBTENCIÓN DEL VECTOR UNITARIO DEL HAZ
        # En nuestro mapeo NED manual:
        # Componente Norte (X) = cos(angulo)
        # Componente Este (Y)  = sin(angulo)
        # Esto funciona porque definimos Norte=X, Este=Y y el ángulo 0 empieza en X.
        vec_beam_north = np.cos(beam_yaw)
        vec_beam_east  = np.sin(beam_yaw)

        # 5. DECODIFICACIÓN
        scan_line = None
        if hasattr(msg, 'encoding') and msg.encoding == "8UC1":
            try: scan_line = np.frombuffer(msg.data, dtype=np.uint8)
            except: pass
        if scan_line is None:
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                scan_line = cv_img.flatten()
            except: continue

        n_pixels = len(scan_line)
        if n_pixels == 0: continue

        meters_per_pixel = SONAR_RANGE / n_pixels
        dist_array = np.arange(n_pixels) * meters_per_pixel
        
        mask = dist_array > BLIND_ZONE
        valid_dist = dist_array[mask]
        valid_vals = scan_line[mask]

        if len(valid_dist) == 0: continue

        # 6. PROYECCIÓN FINAL (Rasterio usa X=Este, Y=Norte)
        px = sensor_east + (vec_beam_east * valid_dist)
        py = sensor_north + (vec_beam_north * valid_dist)

        global_x.extend(px)
        global_y.extend(py)
        intensities.extend(valid_vals)

        stats['ok'] += 1
        if stats['ok'] % 5000 == 0: print(f"   Líneas: {stats['ok']}...", end='\r')

    print(f"\n-> Fin. Líneas procesadas: {stats['ok']}")
    if stats['ok'] == 0: return None, None, None
    return np.array(global_x), np.array(global_y), np.array(intensities)

def rasterize_and_save(x, y, i, res, output_file, crs):
    print(f"-> Rasterizando...")
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_min == x_max: return

    width = int(np.ceil((x_max - x_min) / res))
    height = int(np.ceil((y_max - y_min) / res))
    
    print(f"   Mapa: {width}x{height} px | {x_max-x_min:.1f}m x {y_max-y_min:.1f}m")
    
    col = np.clip(((x - x_min) / res).astype(np.int32), 0, width - 1)
    row = np.clip(((y_max - y) / res).astype(np.int32), 0, height - 1)
    
    flat = row * width + col
    grid_sum = np.zeros(width * height, dtype=np.float32)
    grid_cnt = np.zeros(width * height, dtype=np.float32)
    
    np.add.at(grid_sum, flat, i)
    np.add.at(grid_cnt, flat, 1)
    
    mosaic = np.zeros(width * height, dtype=np.uint8)
    mask = grid_cnt > 0
    mosaic[mask] = (grid_sum[mask] / grid_cnt[mask]).astype(np.uint8)
    mosaic = mosaic.reshape((height, width))

    transform = from_origin(x_min, y_max, res, res)
    with rasterio.open(output_file, 'w', driver='GTiff', height=height, width=width, 
                       count=1, dtype=mosaic.dtype, crs=crs, transform=transform, nodata=0) as dst:
        dst.write(mosaic, 1)
    print(f"-> Guardado: {os.path.abspath(output_file)}")

def main():
    if not os.path.exists(BAG_FILE): 
        print("ERROR: Bag no encontrado"); return
    
    try: bag = rosbag.Bag(BAG_FILE)
    except Exception as e: 
        print(f"ERROR: {e}"); return

    # 1. TFs
    tf_man = TFManager()
    tf_man.load_static_tfs(bag)
    
    # 2. Nav (NED)
    TOPIC_NAV = '/sparus2/navigator/navigation'
    nav_data = get_nav_data(bag, TOPIC_NAV)

    if nav_data:
        # 3. Procesar
        gx, gy, gi = process_sonar_images(bag, nav_data, tf_man)
        if gx is not None:
            rasterize_and_save(gx, gy, gi, MOSAIC_RES, OUTPUT_TIFF, nav_data[3])

    bag.close()

if __name__ == '__main__':
    main()