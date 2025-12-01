import rosbag
import numpy as np
import ros_numpy
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# --- CONFIGURACIÓN ---
BAG_FILE = '/home/uib/derelictes_ws/bagfiles/sparus2_multibeam_2025-11-10-11-31-34_0 1.bag'

SCAN_TOPIC = '/sparus2/norbit_wbms_multibeam/multibeam_scan'
NAV_TOPIC = '/sparus2/navigator/navigation'

# Nombres de salida
FILE_INTERP = 'nube_mb_interpolada.xyz'
FILE_NEAREST = 'nube_mb_sin_interpolar.xyz'

# --- TRANSFORMACIÓN ESTÁTICA (SENSOR -> VEHÍCULO) ---
OFFSET_X, OFFSET_Y, OFFSET_Z = 0.0, 0.0, 0.0
SENSOR_ROLL  = 0.0
SENSOR_PITCH = 0.0
SENSOR_YAW   = np.radians(90)

# Matriz T_S2V
Rx_s = np.array([[1, 0, 0], [0, np.cos(SENSOR_ROLL), -np.sin(SENSOR_ROLL)], [0, np.sin(SENSOR_ROLL), np.cos(SENSOR_ROLL)]])
Ry_s = np.array([[np.cos(SENSOR_PITCH), 0, np.sin(SENSOR_PITCH)], [0, 1, 0], [-np.sin(SENSOR_PITCH), 0, np.cos(SENSOR_PITCH)]])
Rz_s = np.array([[np.cos(SENSOR_YAW), -np.sin(SENSOR_YAW), 0], [np.sin(SENSOR_YAW), np.cos(SENSOR_YAW), 0], [0, 0, 1]])
R_sensor = np.dot(Rz_s, np.dot(Ry_s, Rx_s))

T_S2V = np.identity(4)
T_S2V[:3, :3] = R_sensor
T_S2V[0, 3] = OFFSET_X
T_S2V[1, 3] = OFFSET_Y
T_S2V[2, 3] = OFFSET_Z
# --------------------------------------------------

def get_transform_matrix(position, rotation_matrix):
    """Helper para construir matriz 4x4"""
    T = np.identity(4)
    T[:3, :3] = rotation_matrix
    T[0, 3] = position[0]
    T[1, 3] = position[1]
    T[2, 3] = position[2]
    return T

def main():
    print(f"Abriendo bagfile: {BAG_FILE}...")
    try:
        bag = rosbag.Bag(BAG_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. CARGAR DATOS DE NAVEGACIÓN
    print("-> Indexando navegación...")
    nav_times = []
    nav_positions = []
    nav_quats = [] 

    for topic, msg, t in bag.read_messages(topics=[NAV_TOPIC]):
        nav_times.append(t.to_sec())
        nav_positions.append([msg.position.north, msg.position.east, msg.position.depth])
        
        # Convertir Euler a Quaternión para facilitar manejo
        r = R.from_euler('xyz', [msg.orientation.roll, msg.orientation.pitch, msg.orientation.yaw], degrees=False)
        nav_quats.append(r.as_quat())

    nav_times = np.array(nav_times)
    nav_positions = np.array(nav_positions)
    nav_quats = np.array(nav_quats)

    # 2. CREAR INTERPOLADORES (Para método Interpolado)
    interp_pos = interp1d(nav_times, nav_positions, axis=0, kind='linear', fill_value="extrapolate")
    slerp_rot = Slerp(nav_times, R.from_quat(nav_quats))

    total_scans = bag.get_message_count(SCAN_TOPIC)
    print(f"-> Procesando {total_scans} scans...")

    buffer_interp = []
    buffer_nearest = []
    
    count = 0

    # 3. BUCLE PRINCIPAL
    for topic, scan_msg, t in bag.read_messages(topics=[SCAN_TOPIC]):
        scan_time = t.to_sec()
        
        # Filtros de tiempo
        if scan_time < nav_times[0] or scan_time > nav_times[-1]:
            continue

        # A. Obtener Puntos Locales
        pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(scan_msg)
        mask = np.isfinite(pc_data['x']) & np.isfinite(pc_data['y']) & np.isfinite(pc_data['z'])
        pc_data = pc_data[mask]
        if len(pc_data) == 0: continue
        
        ones = np.ones((len(pc_data), 1))
        points_local = np.column_stack((pc_data['x'], pc_data['y'], pc_data['z'], ones))

        # --- MÉTODO 1: INTERPOLACIÓN (Suave) ---
        try:
            pos_i = interp_pos(scan_time)
            rot_i = slerp_rot(scan_time).as_matrix()
            
            T_world_i = np.dot(get_transform_matrix(pos_i, rot_i), T_S2V)
            pts_world_i = np.dot(T_world_i, points_local.T).T
            
            for p in pts_world_i:
                buffer_interp.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        except:
            pass

        # --- MÉTODO 2: VECINO MÁS CERCANO (Sin Interpolar) ---
        try:
            # Busca el índice del tiempo más cercano
            idx_near = (np.abs(nav_times - scan_time)).argmin()
            
            pos_n = nav_positions[idx_near]
            # Convertimos el cuaternión guardado a matriz
            rot_n = R.from_quat(nav_quats[idx_near]).as_matrix()
            
            T_world_n = np.dot(get_transform_matrix(pos_n, rot_n), T_S2V)
            pts_world_n = np.dot(T_world_n, points_local.T).T

            for p in pts_world_n:
                buffer_nearest.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        except:
            pass
        
        count += 1
        if count % 100 == 0:
            print(f"Procesados: {count}/{total_scans}", end='\r')

    bag.close()
    
    print(f"\n-> Guardando {FILE_INTERP} ({len(buffer_interp)} puntos)...")
    with open(FILE_INTERP, 'w') as f:
        f.writelines(buffer_interp)

    print(f"-> Guardando {FILE_NEAREST} ({len(buffer_nearest)} puntos)...")
    with open(FILE_NEAREST, 'w') as f:
        f.writelines(buffer_nearest)
        
    print("¡Hecho! Compara ambos ficheros en CloudCompare.")

if __name__ == '__main__':
    main()