#!/usr/bin/env python3

import rosbag
import numpy as np
import cv2
import os

BAG_FILE = '/home/uib/bagfiles/cabrera/2025_10_29/16_20_34/sparus2_sidescan_2025-10-29-16-20-34_0.bag'
NAV_TOPIC = '/sparus2/navigator/navigation'
SONAR_RANGE = 30.0
BLIND_ZONE = 0.5
OUTPUT_IMG = 'sss_waterfall_color.png'


def has_image_fields(msg):
    return (
        hasattr(msg, 'data') and
        hasattr(msg, 'header') and
        isinstance(msg.data, (bytes, bytearray)) and
        len(msg.data) > 0
    )

def enhance_data(img_gray):
    """
    Aplica los filtros de realce a una imagen en escala de grises.
    Es mejor aplicar esto ANTES de convertir a color.
    """
    if img_gray.size == 0:
        return img_gray

    img = img_gray.copy()

    # 1. Normalización robusta
    p2, p98 = np.percentile(img, (2, 98))
    # Evitar división por cero
    denom = p98 - p2 if (p98 - p2) > 0 else 1
    img = np.clip((img - p2) * 255.0 / denom, 0, 255).astype(np.uint8)

    # 2. Reducción de speckle
    img = cv2.medianBlur(img, 5)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4. Realce suave
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def main():

    bag = rosbag.Bag(BAG_FILE)

    # ----------------- Navegación (altura) -----------------
    nav_ts = []
    nav_h = []

    print("Leyendo navegación...")
    for _, msg, _ in bag.read_messages(topics=[NAV_TOPIC]):
        if hasattr(msg, 'header') and hasattr(msg, 'altitude'):
            nav_ts.append(msg.header.stamp.to_sec())
            nav_h.append(msg.altitude)

    nav_ts = np.array(nav_ts)
    nav_h = np.array(nav_h)

    if len(nav_ts) == 0:
        print("ERROR: no se encontró altitud en navegación")
        return

    # ----------------- SSS -----------------
    port_lines = []
    star_lines = []

    print("Procesando líneas de sonar...")
    for topic, msg, _ in bag.read_messages():

        if "sidescan" not in topic:
            continue

        if not has_image_fields(msg):
            continue

        ts = msg.header.stamp.to_sec()

        if ts < nav_ts.min() or ts > nav_ts.max():
            continue

        h = np.interp(ts, nav_ts, nav_h)
        if h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8)
        if scan.size < 50:
            continue

        npx = scan.size
        meters_px = SONAR_RANGE / npx

        slant = np.arange(npx) * meters_px
        # Corrección slant-range básica
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0.0))

        valid = ground > BLIND_ZONE
        scan = scan[valid]

        if scan.size < 10:
            continue

        if "port" in topic:
            # Port se invierte para que el nadir quede al centro
            port_lines.append(scan)
        else:
            star_lines.append(scan)

    bag.close()

    # ----------------- Validaciones -----------------
    if len(port_lines) == 0 or len(star_lines) == 0:
        print("ERROR: no hay suficientes líneas port/starboard")
        return

    # ----------------- Normalización de tamaños -----------------
    min_len = min(
        min(len(l) for l in port_lines),
        min(len(l) for l in star_lines)
    )

    # Convertir a arrays numpy
    port_gray = np.array([l[:min_len] for l in port_lines], dtype=np.uint8)
    star_gray = np.array([l[:min_len] for l in star_lines], dtype=np.uint8)

    # Igualar número de filas (pings)
    min_rows = min(port_gray.shape[0], star_gray.shape[0])
    port_gray = port_gray[:min_rows]
    star_gray = star_gray[:min_rows]

    # ----------------- FILTRADO (Antes del color) -----------------
    # Aplicamos el realce sobre la escala de grises para no perder calidad
    print("Aplicando filtros de realce...")
    port_enhanced = enhance_data(port_gray)
    star_enhanced = enhance_data(star_gray)

    # ----------------- COLORIZACIÓN -----------------
    # OpenCV usa formato BGR (Blue, Green, Red)
    rows, cols = port_enhanced.shape
    
    # Crear imagen para PORT (Rojo) -> Canal 2
    port_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    port_color[:, :, 2] = port_enhanced  # Asignar datos al canal ROJO

    # Crear imagen para STARBOARD (Verde) -> Canal 1
    star_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    star_color[:, :, 1] = star_enhanced  # Asignar datos al canal VERDE

    # Crear Nadir (Negro)
    nadir = np.zeros((min_rows, 10, 3), dtype=np.uint8)

    # ----------------- FUSIÓN -----------------
    # Concatenar horizontalmente: [ROJO | NEGRO | VERDE]
    final_img = np.hstack((port_color, nadir, star_color))

    # ----------------- GUARDAR RESULTADO -----------------
    cv2.imwrite(OUTPUT_IMG, final_img)
    print(f"OK -> Imagen guardada en: {os.path.abspath(OUTPUT_IMG)}")
    print(f"Dimensiones: {final_img.shape[1]} x {final_img.shape[0]} px")


if __name__ == "__main__":
    main()