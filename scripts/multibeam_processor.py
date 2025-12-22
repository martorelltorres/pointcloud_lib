#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import ros_numpy
import open3d as o3d
import os
import copy
import tf.transformations as tr 
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def get_transform_matrix(position, rotation_matrix):
    """Construye matriz homogénea 4x4 (Vehículo -> Mundo)"""
    T = np.identity(4)
    T[:3, :3] = rotation_matrix
    T[0, 3] = position[0]
    T[1, 3] = position[1]
    T[2, 3] = position[2]
    return T

def get_static_transform_from_tf(bag_file, parent_frame, child_frame):
    """Intenta leer /tf_static del bagfile para obtener T_S2V automáticamente"""
    rospy.loginfo(f"Buscando transformación estática {parent_frame} -> {child_frame} en bagfile...")
    try:
        bag = rosbag.Bag(bag_file)
        for _, msg, _ in bag.read_messages(topics=['/tf_static', '/tf']):
            for transform in msg.transforms:
                if transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame:
                    tx = transform.transform.translation.x
                    ty = transform.transform.translation.y
                    tz = transform.transform.translation.z
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w
                    
                    T = tr.quaternion_matrix([qx, qy, qz, qw])
                    T[0, 3] = tx
                    T[1, 3] = ty
                    T[2, 3] = tz
                    rospy.loginfo("¡Transformación encontrada en /tf!")
                    return T
        bag.close()
    except Exception:
        pass
    rospy.logwarn("No se encontró TF en el bagfile. Usando parámetros manuales.")
    return None

def save_pcd(pcd, filename, label):
    if pcd is None or len(pcd.points) == 0: return
    rospy.loginfo(f"  -> Guardando {label}: {os.path.basename(filename)} ({len(pcd.points)} pts)...")
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

def main():
    rospy.init_node('multibeam_processor', anonymous=True)
    
    try:
        # --- PARÁMETROS ---
        bag_file = rospy.get_param('~bag_file')
        scan_topic = rospy.get_param('~scan_topic')
        nav_topic = rospy.get_param('~nav_topic')
        output_dir = rospy.get_param('~output_dir')
        
        # Frames TF (Para búsqueda automática)
        base_frame = rospy.get_param('~base_frame_id', 'sparus2/base_link')
        sensor_frame = rospy.get_param('~sensor_frame_id', 'sparus2/multibeam')

        # Fallback manual si no hay TF
        off_x = rospy.get_param('~offset_x', 0.0)
        off_y = rospy.get_param('~offset_y', 0.0)
        off_z = rospy.get_param('~offset_z', 0.0)
        sens_yaw = rospy.get_param('~sensor_yaw', 1.5708) 
        
        # Configuración SLAM y Filtros
        enable_slam = rospy.get_param('~enable_slam', True)
        map_voxel = rospy.get_param('~slam_map_voxel', 0.5)
        gicp_dist = rospy.get_param('~slam_gicp_dist', 2.0)
        sor_k = rospy.get_param('~sor_k', 50)
        sor_std = rospy.get_param('~sor_std', 1.0)
        voxel_size = rospy.get_param('~voxel_size', 0.1)

    except KeyError as e:
        rospy.logerr(f"Falta parámetro: {e}")
        return

    # --- 1. OBTENER MATRIZ T_S2V (SENSOR -> VEHICULO) ---
    # Intento 1: Leer del bagfile (/tf_static)
    T_S2V = get_static_transform_from_tf(bag_file, base_frame, sensor_frame)

    # Intento 2: Usar parámetros manuales
    if T_S2V is None:
        rospy.loginfo("Usando parámetros manuales para T_S2V...")
        Rz = np.array([[np.cos(sens_yaw), -np.sin(sens_yaw), 0], [np.sin(sens_yaw), np.cos(sens_yaw), 0], [0, 0, 1]])
        T_S2V = np.identity(4)
        T_S2V[:3, :3] = Rz
        T_S2V[0:3, 3] = [off_x, off_y, off_z]
    
    # --- 2. CARGAR NAVEGACIÓN (Roll/Pitch/Yaw del AUV) ---
    rospy.loginfo("Indexando navegación y orientación del AUV...")
    bag = rosbag.Bag(bag_file)
    nav_times, nav_pos, nav_quats = [], [], []
    
    for _, msg, t in bag.read_messages(topics=[nav_topic]):
        nav_times.append(t.to_sec())
        nav_pos.append([msg.position.north, msg.position.east, msg.position.depth])
        # Aquí capturamos Roll, Pitch, Yaw dinámicos del AUV
        r = R.from_euler('xyz', [msg.orientation.roll, msg.orientation.pitch, msg.orientation.yaw], degrees=False)
        nav_quats.append(r.as_quat())

    nav_times = np.array(nav_times)
    interp_pos = interp1d(nav_times, np.array(nav_pos), axis=0, kind='linear', fill_value="extrapolate")
    slerp_rot = Slerp(nav_times, R.from_quat(np.array(nav_quats)))

    # --- 3. PROCESAMIENTO ---
    buffer_odom_raw = []
    buffer_slam = []
    global_map = o3d.geometry.PointCloud()
    total = bag.get_message_count(scan_topic)
    cnt = 0
    
    rospy.loginfo(f"Procesando {total} scans...")

    for topic, scan_msg, t in bag.read_messages(topics=[scan_topic]):
        t_sec = t.to_sec()
        if t_sec < nav_times[0] or t_sec > nav_times[-1]: continue

        try:
            # A. Puntos en el marco del SENSOR (Multibeam Frame)
            pc = ros_numpy.point_cloud2.pointcloud2_to_array(scan_msg)
            mask = np.isfinite(pc['x'])
            pc = pc[mask]
            if len(pc) < 10: continue

            xyz_local = np.column_stack((pc['x'], pc['y'], pc['z'])).astype(np.float64)
            pcd_local = o3d.geometry.PointCloud()
            pcd_local.points = o3d.utility.Vector3dVector(xyz_local)
            
            # B. Transformar al marco del VEHÍCULO (Base Link) usando T_S2V fija
            pcd_vehicle = pcd_local.transform(T_S2V)

            # C. Transformar al marco MUNDO (World NED) usando Pose Dinámica del AUV
            pos_pred = interp_pos(t_sec)
            rot_pred = slerp_rot(t_sec).as_matrix()
            
            T_odom = get_transform_matrix(pos_pred, rot_pred)

            # --- BUFFER 1: ODOMETRÍA PURA ---
            pcd_odom_frame = copy.deepcopy(pcd_vehicle).transform(T_odom)
            
            # MODIFICACIÓN: Invertir Z para el guardado (Profundidad -> Elevación)
            pts_odom_save = np.asarray(pcd_odom_frame.points)
            pts_odom_save[:, 2] *= -1.0 
            buffer_odom_raw.append(pts_odom_save)

            # --- BUFFER 2: SLAM (Corrección sobre la odometría) ---
            T_corrected = T_odom 

            if enable_slam and len(global_map.points) > 500:
                source = pcd_vehicle 
                try:
                    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
                    reg = o3d.pipelines.registration.registration_generalized_icp(
                        source, global_map, 
                        max_correspondence_distance=gicp_dist,
                        init=T_odom,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15) 
                    )
                    T_corrected = reg.transformation
                except Exception:
                    pass
            
            pcd_slam_frame = copy.deepcopy(pcd_vehicle).transform(T_corrected)
            
            # MODIFICACIÓN: Invertir Z SOLO para el buffer de guardado, no para el mapa
            pts_slam_numpy = np.asarray(pcd_slam_frame.points)
            
            # Copia para guardar con Z negativa
            pts_slam_save = pts_slam_numpy.copy()
            pts_slam_save[:, 2] *= -1.0
            buffer_slam.append(pts_slam_save)

            if enable_slam:
                # Al mapa global lo añadimos con la Z original (Positiva/NED) para que las matemáticas cuadren
                global_map += pcd_slam_frame
                if cnt % 20 == 0: 
                    global_map = global_map.voxel_down_sample(voxel_size=map_voxel)
                    global_map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=map_voxel*2, max_nn=30))

        except Exception:
            pass
        
        cnt += 1
        if cnt % 100 == 0:
            print(f"Procesando: {cnt}/{total}", end='\r')

    bag.close()
    
    # --- GUARDADO ---
    # 1. RAW Odometría
    if buffer_odom_raw:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(buffer_odom_raw))
        save_pcd(pcd, output_dir + "1_mb_RAW_odom.xyz", "RAW Odometry (Z-)")
        
        # 2. SOR
        pcd_sor, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_k, std_ratio=sor_std)
        save_pcd(pcd_sor, output_dir + "2_mb_SOR.xyz", "SOR Filtered (Z-)")

        # 3. VOXEL
        pcd_vox = pcd.voxel_down_sample(voxel_size=voxel_size)
        save_pcd(pcd_vox, output_dir + "3_mb_VOXEL.xyz", "Voxel Grid (Z-)")
    
    # 4. SLAM
    if buffer_slam:
        pcd_slam = o3d.geometry.PointCloud()
        pcd_slam.points = o3d.utility.Vector3dVector(np.vstack(buffer_slam))
        pcd_slam = pcd_slam.voxel_down_sample(voxel_size=voxel_size)
        save_pcd(pcd_slam, output_dir + "4_mb_SLAM_corrected.xyz", "SLAM Corrected (Z-)")

if __name__ == '__main__':
    main()