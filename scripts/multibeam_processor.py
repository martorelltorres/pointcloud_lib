#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import ros_numpy
import os
from nav_msgs.msg import Odometry
from cola2_msgs.msg import NavSts 
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def get_transform_matrix(position, rotation_matrix):
    """Helper para construir matriz 4x4"""
    T = np.identity(4)
    T[:3, :3] = rotation_matrix
    T[0, 3] = position[0]
    T[1, 3] = position[1]
    T[2, 3] = position[2]
    return T

def main():
    rospy.init_node('multibeam_processor', anonymous=True)
    
    # --- 1. OBTENER PARÁMETROS DE ROS ---
    try:
        # Rutas y Topics
        bag_file = rospy.get_param('~bag_file')
        scan_topic = rospy.get_param('~scan_topic')
        nav_topic = rospy.get_param('~nav_topic')
        file_interp = rospy.get_param('~file_interp')
        file_nearest = rospy.get_param('~file_nearest')
        
        # Transformación Estática (T_S2V)
        offset_x = rospy.get_param('~offset_x', 0.0)
        offset_y = rospy.get_param('~offset_y', 0.0)
        offset_z = rospy.get_param('~offset_z', 0.0)
        sensor_roll = rospy.get_param('~sensor_roll', 0.0)
        sensor_pitch = rospy.get_param('~sensor_pitch', 0.0)
        sensor_yaw = rospy.get_param('~sensor_yaw', np.radians(90)) # Default 90 deg
        
    except KeyError as e:
        rospy.logerr(f"Falta el parámetro de configuración crucial: {e}. Abortando.")
        return

    # --- 2. CONSTRUCCIÓN DE LA MATRIZ T_S2V CON PARÁMETROS ---
    Rx_s = np.array([[1, 0, 0], [0, np.cos(sensor_roll), -np.sin(sensor_roll)], [0, np.sin(sensor_roll), np.cos(sensor_roll)]])
    Ry_s = np.array([[np.cos(sensor_pitch), 0, np.sin(sensor_pitch)], [0, 1, 0], [-np.sin(sensor_pitch), 0, np.cos(sensor_pitch)]])
    Rz_s = np.array([[np.cos(sensor_yaw), -np.sin(sensor_yaw), 0], [np.sin(sensor_yaw), np.cos(sensor_yaw), 0], [0, 0, 1]])
    R_sensor = np.dot(Rz_s, np.dot(Ry_s, Rx_s))

    T_S2V = np.identity(4)
    T_S2V[:3, :3] = R_sensor
    T_S2V[0, 3] = offset_x
    T_S2V[1, 3] = offset_y
    T_S2V[2, 3] = offset_z
    # ----------------------------------------------------------------------


    print(f"Abriendo bagfile: {bag_file}...")
    try:
        bag = rosbag.Bag(bag_file)
    except Exception as e:
        rospy.logerr(f"Error al abrir el bagfile: {e}")
        return

    # 3. CARGAR DATOS DE NAVEGACIÓN
    print("-> Indexando navegación...")
    nav_times = []
    nav_positions = []
    nav_quats = [] 

    for topic, msg, t in bag.read_messages(topics=[nav_topic]):
        nav_times.append(t.to_sec())
        nav_positions.append([msg.position.north, msg.position.east, msg.position.depth])
        
        r = R.from_euler('xyz', [msg.orientation.roll, msg.orientation.pitch, msg.orientation.yaw], degrees=False)
        nav_quats.append(r.as_quat())

    nav_times = np.array(nav_times)
    nav_positions = np.array(nav_positions)
    nav_quats = np.array(nav_quats)

    # 4. CREAR INTERPOLADORES
    interp_pos = interp1d(nav_times, nav_positions, axis=0, kind='linear', fill_value="extrapolate")
    slerp_rot = Slerp(nav_times, R.from_quat(nav_quats))

    total_scans = bag.get_message_count(scan_topic)
    rospy.loginfo(f"Datos de NAV indexados. Procesando {total_scans} scans...")

    buffer_interp = []
    buffer_nearest = []
    
    count = 0

    # 5. BUCLE PRINCIPAL DE PROCESAMIENTO
    for topic, scan_msg, t in bag.read_messages(topics=[scan_topic]):
        scan_time = t.to_sec()
        
        # Filtros de tiempo
        if scan_time < nav_times[0] or scan_time > nav_times[-1]:
            continue

        # A. Obtener Puntos Locales
        try:
            pc_data = ros_numpy.point_cloud2.pointcloud2_to_array(scan_msg)
            mask = np.isfinite(pc_data['x']) & np.isfinite(pc_data['y']) & np.isfinite(pc_data['z'])
            pc_data = pc_data[mask]
        except Exception as e:
            rospy.logwarn(f"Error al procesar PointCloud2: {e}")
            continue

        if len(pc_data) == 0: continue
        
        ones = np.ones((len(pc_data), 1))
        points_local = np.column_stack((pc_data['x'], pc_data['y'], pc_data['z'], ones))

        # --- MÉTODO 1: INTERPOLACIÓN (SLERP) ---
        try:
            pos_i = interp_pos(scan_time)
            rot_i = slerp_rot(scan_time).as_matrix()
            T_world_i = np.dot(get_transform_matrix(pos_i, rot_i), T_S2V)
            pts_world_i = np.dot(T_world_i, points_local.T).T
            
            for p in pts_world_i:
                buffer_interp.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        except Exception as e:
            rospy.logwarn(f"Fallo en interpolación: {e}")

        # --- MÉTODO 2: VECINO MÁS CERCANO (Nearest Neighbor) ---
        try:
            idx_near = (np.abs(nav_times - scan_time)).argmin()
            pos_n = nav_positions[idx_near]
            rot_n = R.from_quat(nav_quats[idx_near]).as_matrix()
            
            T_world_n = np.dot(get_transform_matrix(pos_n, rot_n), T_S2V)
            pts_world_n = np.dot(T_world_n, points_local.T).T

            for p in pts_world_n:
                buffer_nearest.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        except Exception as e:
            rospy.logwarn(f"Fallo en vecino cercano: {e}")
        
        count += 1
        if count % 100 == 0:
            rospy.loginfo(f"Procesados: {count}/{total_scans} scans")

    bag.close()
    
    # 6. Guardar Archivos
    rospy.loginfo(f"Guardando {file_interp} ({len(buffer_interp)} puntos)...")
    with open(file_interp, 'w') as f:
        f.writelines(buffer_interp)

    rospy.loginfo(f"Guardando {file_nearest} ({len(buffer_nearest)} puntos)...")
    with open(file_nearest, 'w') as f:
        f.writelines(buffer_nearest)
        
    rospy.loginfo("Procesamiento finalizado. Archivos guardados.")

if __name__ == '__main__':
    main()