import os
import imageio
import json
import numpy as np
from matplotlib import pyplot as plt
from data_loader.parseTrackletXML import parseXML


_sem2label = {
    'Misc': -1,
    'Car': 0,
    'Van': 0,
    'Truck': 2,
    'Tram': 3,
    'Pedestrian': 4
}

camera_ls = [2, 3]


def kitti_string_to_float(str):
    return float(str.split('e')[0]) * 10**int(str.split('e')[1])


def get_scene_objects(basedir):

    kitti_obj_meta = {}
    tracklet_file = os.path.join(basedir, 'tracklet_labels.xml')
    tracklets = parseXML(tracklet_file)

    for i, object in enumerate(tracklets):
        # if end > start_frame and start < end_frame:
        id = i+1
        # height, width, length --> length, height, width
        dim = object.size[[2, 0, 1]]
        if object.objectType in _sem2label:
            meta_obj = [id, id, _sem2label[object.objectType], dim]
            kitti_obj_meta[id] = meta_obj

    return kitti_obj_meta


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0., 0., 0., 1.]])])


def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot


def get_obj_pose_tracking(tracklet_path, poses_imu_tracking, calibrations, selected_frames):

    def roty_matrix(roty):
        c = np.cos(roty)
        s = np.sin(roty)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    velo2cam = calibrations['Tr_velo2cam']
    imu2velo = calibrations['Tr_imu2velo']
    cam2velo = invert_transformation(velo2cam[:3, :3], velo2cam[:3, 3])
    velo2imu = invert_transformation(imu2velo[:3, :3], imu2velo[:3, 3])

    objects_meta_kitti = {}
    objects_meta = {}
    tracklets_ls = []

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    f = open(tracklet_path)
    tracklets_str = f.read().splitlines()

    n_scene_frames = len(poses_imu_tracking)
    n_obj_in_frame = np.zeros(n_scene_frames)

    # Metadata for all objects in the scene
    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        if float(tracklet[1]) < 0:
            continue
        id = tracklet[1]
        if tracklet[2] in _sem2label:
            type = _sem2label[tracklet[2]]
            if not int(id) in objects_meta_kitti:
                length = tracklet[12]
                height = tracklet[10]
                width = tracklet[11]
                objects_meta_kitti[int(id)] = np.array([float(id), type, length, height, width])

            tr_array = np.concatenate([np.array(tracklet[:2]).astype(np.float), np.array([type]), np.array(tracklet[3:]).astype(np.float)])
            tracklets_ls.append(tr_array)
            n_obj_in_frame[int(tracklet[0])] += 1

    # Objects in each frame
    tracklets_array = np.array(tracklets_ls)

    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame+1].max())
    visible_objects = np.ones([(end_frame - start_frame + 1) * 2, max_obj_per_frame, 14]) * -1.

    for tracklet in tracklets_array:
        frame_no = tracklet[0]
        if start_frame <= frame_no <= end_frame:

            obj_id = tracklet[1]
            frame_id = np.array([frame_no])
            id_int = int(obj_id)
            obj_type = np.array([objects_meta_kitti[id_int][1]])
            dim = objects_meta_kitti[id_int][-3:].astype(np.float32)

            if id_int not in objects_meta:
                objects_meta[id_int] = np.concatenate([np.array([id_int]).astype(np.float32),
                                                       objects_meta_kitti[id_int][2:].astype(np.float64),
                                                       np.array([objects_meta_kitti[id_int][1]]).astype(np.float64)])

            pose = tracklet[-4:]

            obj_pose_c = np.eye(4)
            obj_pose_c[:3, 3] = pose[:3]
            roty = pose[3]
            obj_pose_c[:3, :3] = roty_matrix(roty)

            obj_pose_imu = np.matmul(velo2imu, np.matmul(cam2velo, obj_pose_c))

            pose_imu_w_frame_i = poses_imu_tracking[int(frame_id)]

            pose_obj_w_i = np.matmul(pose_imu_w_frame_i, obj_pose_imu)

            yaw_aprox = -np.arctan2(pose_obj_w_i[1, 0], pose_obj_w_i[0, 0])

            # TODO: Change if necessary
            is_moving = 1.

            pose_3d = np.array([pose_obj_w_i[0, 3],
                                pose_obj_w_i[1, 3],
                                pose_obj_w_i[2, 3],
                                yaw_aprox, 0, 0, is_moving])

            for j, cam in enumerate(camera_ls):
                cam = np.array(cam).astype(np.float32)[None]
                obj = np.concatenate([frame_id, cam, np.array([obj_id]), obj_type, dim, pose_3d])
                frame_cam_id = (int(frame_no) - start_frame) + j*(end_frame+1 - start_frame)
                obj_column = np.argwhere(visible_objects[frame_cam_id, :, 0] < 0).min()
                visible_objects[frame_cam_id, obj_column] = obj

    # Remove not moving objects
    print('Removing non moving objects')
    obj_to_del = []
    for key, values in objects_meta.items():
        all_obj_poses = np.where(visible_objects[:, :, 2] == key)
        if len(all_obj_poses[0]) > 0 and values[4] != 4.:
            frame_intervall = all_obj_poses[0][[0, -1]]
            y = all_obj_poses[1][[0, -1]]
            obj_poses = visible_objects[[frame_intervall, y]][:, 7:10]
            distance = np.linalg.norm(obj_poses[1] - obj_poses[0])
            print(distance)
            if distance < 0.5:
                print('Removed:', key)
                obj_to_del.append(key)
                visible_objects[all_obj_poses] = np.ones(14) * -1.

    for key in obj_to_del:
        del objects_meta[key]

    return visible_objects, objects_meta


def get_obj_poses(basedir, poses_velo_w, calibrations, kitti_obj_meta, selected_frames):

    def rotation_yaw(yaw):
        c_y = np.cos(yaw)
        s_y = np.sin(yaw)
        return np.array([[c_y, -s_y, 0],
                         [s_y, c_y, 0],
                         [0, 0, 1]])

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    objects_meta = {}
    visible_objects = np.ones([(end_frame - start_frame+1)*2, len(kitti_obj_meta)+1, 14]) * -1.

    tracklet_file = os.path.join(basedir, 'tracklet_labels.xml')
    tracklets = parseXML(tracklet_file)

    imu2velo, velo2cam, c2leftRGB, c2rightRGB, _ = calibrations

    for i, object in enumerate(tracklets):
        start = object.firstFrame
        end = object.firstFrame + object.nFrames
        id = i + 1
        id_int = np.array(id).astype(np.float32)[None]
        if start <= end_frame and end >= start_frame:
            for frame_no in range(max(start_frame, start), min(end_frame+1, end)):
                obj_step = frame_no-start
                frame_id = np.array(frame_no).astype(np.float32)[None]
                # Velodyne pose at this step
                pose_velo_w_i = poses_velo_w[frame_no]

                for j, cam in enumerate(camera_ls):
                    cam = np.array(cam).astype(np.float32)[None]

                    # Create meta dict for this specific sequence from scene metadata
                    if id not in objects_meta:
                        objects_meta[id] = np.concatenate([np.array([id]).astype(np.float32),
                                                           np.zeros([19]), kitti_obj_meta[id][3]])

                    obj_type = np.array(kitti_obj_meta[id][2]).astype(np.float32)[None]
                    # length, height, width --> length, width,  height
                    dim = np.array(kitti_obj_meta[id][3])[[0, 2, 1]]

                    t_obj = np.array(object.trans[obj_step]) # + np.array([0., 0., 1.]) * dim/2
                    roty = np.array(object.rots[obj_step][2]) #+ np.pi/2
                    rot_obj = rotation_yaw(roty)
                    # t_mid = np.matmul(rot_obj, np.array([0., 1., 1.]) * dim/2)
                    # t_obj += t_mid

                    obj_pose_v = np.concatenate([np.concatenate([rot_obj, t_obj[:, None]], axis=1), np.array([[0.,0.,0.,1.]])])

                    # Velo to world transformation
                    pose_obj_w_i = np.matmul(pose_velo_w_i, obj_pose_v)

                    yaw_aprox = -np.arctan2(pose_obj_w_i[1, 0], pose_obj_w_i[0, 0])

                    # TODO: Change if necessary
                    is_moving = 1.

                    pose_3d = np.array([pose_obj_w_i[0, 3],
                                        pose_obj_w_i[1, 3],
                                        pose_obj_w_i[2, 3],
                                        yaw_aprox, 0, 0, is_moving])

                    obj = np.concatenate([frame_id, cam, id_int, obj_type, dim, pose_3d])

                    visible_objects[(frame_no-start_frame)*2+j, int(id)] = obj

    object_ids = []
    for k in objects_meta.keys():
        object_ids.append(k)

    visible_objects = visible_objects[..., np.array(object_ids), :]

    return visible_objects, objects_meta


def calib_from_txt(calibration_path):

    c2c = []

    f = open(os.path.join(calibration_path, 'calib_cam_to_cam.txt'), "r")
    cam_to_cam_str = f.read()
    [left_cam, right_cam] = cam_to_cam_str.split('S_02: ')[1].split('S_03: ')
    cam_to_cam_ls = [left_cam, right_cam]

    for i, cam_str in enumerate(cam_to_cam_ls):
        r_str, t_str = cam_str.split('R_0' + str(i + 2) + ': ')[1].split('\nT_0' + str(i + 2) + ': ')
        t_str = t_str.split('\n')[0]
        R = np.array([kitti_string_to_float(r) for r in r_str.split(' ')])
        R = np.reshape(R, [3, 3])
        t = np.array([kitti_string_to_float(t) for t in t_str.split(' ')])
        Tr = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])


        t_str_rect, s_rect_part = cam_str.split('\nT_0' + str(i + 2) + ': ')[1].split('\nS_rect_0' + str(i + 2) + ': ')
        s_rect_str, r_rect_part = s_rect_part.split('\nR_rect_0' + str(i + 2) + ': ')
        r_rect_str = r_rect_part.split('\nP_rect_0' + str(i + 2) + ': ')[0]
        R_rect = np.array([kitti_string_to_float(r) for r in r_rect_str.split(' ')])
        R_rect = np.reshape(R_rect, [3, 3])
        t_rect = np.array([kitti_string_to_float(t) for t in t_str_rect.split(' ')])
        Tr_rect = np.concatenate([np.concatenate([R_rect, t_rect[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])


        c2c.append(Tr_rect)

    c2leftRGB = c2c[0]
    c2rightRGB = c2c[1]

    f = open(os.path.join(calibration_path, 'calib_velo_to_cam.txt'), "r")
    velo_to_cam_str = f.read()
    r_str, t_str = velo_to_cam_str.split('R: ')[1].split('\nT: ')
    t_str = t_str.split('\n')[0]
    R = np.array([kitti_string_to_float(r) for r in r_str.split(' ')])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(' ')])
    v2c = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])

    f = open(os.path.join(calibration_path, 'calib_imu_to_velo.txt'), "r")
    imu_to_velo_str = f.read()
    r_str, t_str = imu_to_velo_str.split('R: ')[1].split('\nT: ')
    R = np.array([kitti_string_to_float(r) for r in r_str.split(' ')])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(' ')])
    imu2v = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])

    focal = kitti_string_to_float(left_cam.split('P_rect_02: ')[1].split()[0])

    return imu2v, v2c, c2leftRGB, c2rightRGB, focal


def tracking_calib_from_txt(calibration_path):
    f = open(calibration_path)
    calib_str = f.read().splitlines()
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0., 0., 0., 1.]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0., 0., 0., 1.]])], axis=0)

    return {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3, 'Tr_cam2camrect': Tr_cam2camrect,
            'Tr_velo2cam': Tr_velo2cam, 'Tr_imu2velo': Tr_imu2velo}


def get_poses_calibration(basedir, oxts_path_tracking=None, selected_frames=None):

    def oxts_to_pose(oxts):
        poses = []

        def latlon_to_mercator(lat, lon, s):
            r = 6378137.0
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        if selected_frames == None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])
                # pose_0_inv = np.linalg.inv(pose_i)

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, 'oxts/data')
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    ### Tracking oxts
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    return poses, calibrations, focal


def get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, selected_frames, scene_no=None, exp=False):
    camera_poses = []

    opengl2kitti = np.array([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7) ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5) ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9) ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3) ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6) ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75) ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        #roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration['Tr_cam2camrect']
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = invert_transformation(Tr_cam2camrect[:3, :3], Tr_cam2camrect[:3, 3])
    Tr_velo2cam = tracking_calibration['Tr_velo2cam']
    Tr_cam2velo = invert_transformation(Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3])

    camera_poses_imu = []
    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration['Tr_camrect2cam0' + str(cam)]
        Tr_cam_i2camrect = invert_transformation(Tr_camrect2cam_i[:3, :3], Tr_camrect2cam_i[:3, 3])
        # transform camera axis from kitti to opengl for nerf:
        cam_i_camrect = np.matmul(Tr_cam_i2camrect, opengl2kitti)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, cam_i_camrect)
        cam_i_velo = np.matmul(Tr_cam2velo, cam_i_cam0)

        cam_i_w = np.matmul(poses_velo_w_tracking, cam_i_velo)
        camera_poses_imu.append(cam_i_w)


    for i, cam in enumerate(camera_ls):
        for frame_no in range(start_frame, end_frame + 1):
            camera_poses.append(camera_poses_imu[i][frame_no])

    return np.array(camera_poses)


def get_camera_poses(poses_velo_w, calibrations, selected_frames):

    imu2velo, velo2c0, c02leftRGB, c02rightRGB, _ = calibrations
    c02cam = [c02leftRGB, c02rightRGB]

    camera_poses = []

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    c02velo = invert_transformation(velo2c0[:3, :3], velo2c0[:3, 3])

    # np.concatenate([np.concatenate([v2c[:3, :3].T, v2c[:3, 3][:, None]], axis=1), v2c[3, :][None]])
    # poses_v_w = np.matmul(v2imu, poses_velo_w)

    opengl2kitti = np.array([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

    cam2c0 = []
    for cam_transform in c02cam:
        cam_i_2c0 = invert_transformation(cam_transform[:3, :3], cam_transform[:3, 3])
        # transform camera axis from kitti to opengl for nerf:
        cam_i_2c0 = np.matmul(cam_i_2c0, opengl2kitti)

        cam2c0.append(cam_i_2c0)

    c_left_v = np.matmul(c02velo, cam2c0[0])
    c_right_v = np.matmul(c02velo, cam2c0[1])
    c_left_w = np.matmul(poses_velo_w, c_left_v)
    c_right_w = np.matmul(poses_velo_w, c_right_v)
    poses_cam_lr_rgb = [c_left_w, c_right_w]

    for frame_no in range(start_frame, end_frame+1):
        for i, cam in enumerate(camera_ls):
            camera_poses.append(poses_cam_lr_rgb[i][frame_no])

    return np.array(camera_poses)


def get_scene_images(basedir, selected_frames):
    [start_frame, end_frame] = selected_frames
    imgs = []

    left_img_path = 'image_02/data'
    right_img_path = 'image_03/data'

    for frame_no in range(len(os.listdir(os.path.join(basedir, left_img_path)))):
        if start_frame <= frame_no <= end_frame:
            for pth in [left_img_path, right_img_path]:
                frame_dir = os.path.join(basedir, pth)
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                imgs.append(imageio.imread(fname))

    imgs = (np.maximum(np.minimum(np.array(imgs), 255), 0) / 255.).astype(np.float32)

    return imgs


def get_scene_images_tracking(tracking_path, sequence, selected_frames):
    [start_frame, end_frame] = selected_frames
    imgs = []

    left_img_path = os.path.join(os.path.join(tracking_path, 'image_02'), sequence)
    right_img_path = os.path.join(os.path.join(tracking_path, 'image_03'), sequence)

    for frame_dir in [left_img_path, right_img_path]:
        for frame_no in range(len(os.listdir(left_img_path))):
            if start_frame <= frame_no <= end_frame:
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                imgs.append(imageio.imread(fname))

    imgs = (np.maximum(np.minimum(np.array(imgs), 255), 0) / 255.).astype(np.float32)

    return imgs


def load_kitti_data(basedir, selected_frames=None, use_obj=True, row_id=False, remove=-1, use_time=False, exp=False):
    """loads vkitti data

    Args:
        basedir: directory like "kitti/2011_09_26/2011_09_26_drive_0018_sync"
        selected_frames: [first_frame, last_frame]
        use_obj: bool
        row_id: bool

    Returns:
        imgs: [n_frames, h, w, 3]
        instance_segm: [n_frames, h, w]
        poses: [n_frames, 4, 4]
        frame_id: [n_frames]: [frame, cam, 0]
        render_poses: [n_test_frames, 4, 4]
        hwf: [H, W, focal]
        i_split: [[train_split], [validation_split], [test_split]]
        visible_objects: [n_frames, n_obj, 23]
        object_meta: dictionary with metadata for each object with track_id as key
        render_objects: [n_test_frames, n_obj, 23]
        bboxes: 2D bounding boxes in the images stored for each of n_frames
    """

    if selected_frames is None:
        selected_frames = [0, 0]

    kitti2vkitti = np.array([[1., 0., 0., 0.],
                             [0., 0., -1., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 0., 1.]])

    # basedir = '/home/julian/workspace/kitti_data/2011_09_26/2011_09_26_drive_0056_sync'
    # basedir = '/home/julian/workspace/kitti_data/2011_09_26/2011_09_26_drive_0018_sync'

    #########################################################################################
    #### Tracking Dataset ####
    dataset = 'tracking'
    kitti_scene_no = int(basedir[-4:])

    images_ls = []
    poses_ls = []
    visible_objects_ls = []
    objects_meta_ls = []

    if dataset == 'tracking':
        sequence = basedir[-4:]
        tracking_path = basedir[:-13]
        calibration_path = os.path.join(os.path.join(tracking_path, 'calib'), sequence+'.txt')
        oxts_path_tracking = os.path.join(os.path.join(tracking_path, 'oxts'), sequence+'.txt')
        tracklet_path = os.path.join(os.path.join(tracking_path, 'label_02'), sequence+'.txt')

        tracking_calibration = tracking_calib_from_txt(calibration_path)
        focal = tracking_calibration['P2'][0, 0]
        if exp:
            print('Experimental Coordinate Trafo')
            poses_imu_w_tracking, _, _ = get_poses_calibration(basedir, oxts_path_tracking, selected_frames)
        else:
            poses_imu_w_tracking, _, _ = get_poses_calibration(basedir, oxts_path_tracking)

        tr_imu2velo = tracking_calibration['Tr_imu2velo']
        tr_velo2imu = invert_transformation(tr_imu2velo[:3, :3], tr_imu2velo[:3, 3])
        poses_velo_w_tracking = np.matmul(poses_imu_w_tracking, tr_velo2imu)

        # Get camera Poses
        for cam_i in range(2):
            transformation = np.eye(4)
            projection = tracking_calibration['P'+str(cam_i+2)]
            K_inv = np.linalg.inv(projection[:3, :3])
            R_t = projection[:3, 3]

            t_crect2c = np.matmul(K_inv, R_t)
            # t_crect2c = 1./projection[[0, 1, 2],[0, 1, 2]] * projection[:, 3]
            transformation[:3, 3] = t_crect2c
            tracking_calibration['Tr_camrect2cam0'+str(cam_i+2)] = transformation

        for seq_i, sequ in enumerate(selected_frames[0]):
            sequ_frames = [sequ, selected_frames[1][seq_i]]

            cam_poses_tracking = get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, sequ_frames,
                                                           kitti_scene_no, exp)

            # Get Object poses
            #
            visible_objects, objects_meta = get_obj_pose_tracking(tracklet_path, poses_imu_w_tracking,
                                                                  tracking_calibration, sequ_frames)

            # Align Axis with vkitti axis
            poses = np.matmul(kitti2vkitti, cam_poses_tracking)
            visible_objects[:, :, [9]] *= -1
            visible_objects[:, :, [7, 8, 9]] = visible_objects[:, :, [7, 9, 8]]

            images = get_scene_images_tracking(tracking_path, sequence, sequ_frames)

            images_ls.append(images)
            poses_ls.append(poses)
            visible_objects_ls.append(visible_objects)
            objects_meta_ls.append(objects_meta)

            kitti_obj_metadata = None
    ###########################################################################################

    ###########################################################################################
    ### Synced and rectified Data
    # basedir = '/home/julian/workspace/kitti_data/2011_09_26/2011_09_26_drive_0056_sync'
    # basedir = '/home/julian/workspace/kitti_data/2011_09_26/2011_09_26_drive_0018_sync'

    else:
        for seq_i, sequ in enumerate(selected_frames[0]):
            sequ_frames = [sequ, selected_frames[1][seq_i]]

            poses_imu_w, calibrations, focal = get_poses_calibration(basedir, oxts_path_tracking=None)

            imu2velo, velo2cam, c2leftRGB, c2rightRGB, _ = calibrations

            velo2imu = invert_transformation(calibrations[0][:3, :3], calibrations[0][:3, 3])
            poses_velo_w = np.matmul(poses_imu_w, velo2imu)

            cam_poses = get_camera_poses(poses_velo_w, calibrations, sequ_frames)

            kitti_obj_metadata = get_scene_objects(basedir)

            visible_objects, objects_meta = get_obj_poses(basedir, poses_velo_w, calibrations, kitti_obj_metadata, sequ_frames)

            # Align Axis with vkitti axis
            poses = np.matmul(kitti2vkitti, cam_poses)
            visible_objects[:, :, [9]] *= -1
            visible_objects[:, :, [7, 8, 9]] = visible_objects[:, :, [7, 9, 8]]

            images = get_scene_images(basedir, sequ_frames)

            images_ls.append(images)
            poses_ls.append(poses)
            visible_objects_ls.append(visible_objects)
            objects_meta_ls.append(objects_meta)

    # Debug Velo to IMU poses
    # plt.figure()
    # plt.axis('equal')
    # plt.scatter(poses_imu_w[:, 0, 3], poses_imu_w[:, 1, 3], color='orange')
    # plt.scatter(poses_velo_w[:, 0, 3], poses_velo_w[:, 1, 3], color='black')

    # Debug object poses
    # t_obj = np.reshape(visible_objects[:, :, [7, 8, 9]], [-1, 3])
    # yaw_obj = np.reshape(visible_objects[:, :, [10]], [-1, 1])
    # plt.scatter(t_obj[:, 0], t_obj[:, 1], color='green')
    # for i in range(len(yaw_obj)):
    #     obj_axis = np.matmul(get_rotation(0, 0, -yaw_obj[i][0]), np.eye(3)*3)
    #     plt.arrow(t_obj[i, 0], t_obj[i, 1], obj_axis[0, 0], obj_axis[1, 0], color='black')
    #     plt.arrow(t_obj[i, 0], t_obj[i, 1], obj_axis[0, 1], obj_axis[1, 1], color='red')

    # poses_velo_w_kitti = np.array(poses_velo_w)
    # poses_velo_w = np.matmul(kitti2vkitti, poses_velo_w)

    # x_plt = 0
    # y_plt = 1
    # plt.scatter(poses[:, x_plt, 3], poses[:, y_plt, 3], color='blue')
    # plt.axis('equal')
    # for plt_no in range(poses.shape[0]):
    #     plt.arrow(poses[plt_no, x_plt, 3], poses[plt_no, y_plt, 3], poses[plt_no, x_plt, 0], poses[plt_no, y_plt, 0], color='green')
    #     plt.arrow(poses[plt_no, x_plt, 3], poses[plt_no, y_plt, 3], poses[plt_no, x_plt, 1], poses[plt_no, y_plt, 1])
    #     plt.arrow(poses[plt_no, x_plt, 3], poses[plt_no, y_plt, 3], poses[plt_no, x_plt, 2], poses[plt_no, y_plt, 2], color='red')

    # plt.figure()
    # plt.scatter(-cam_poses[0, 1, 3], cam_poses[0, 2, 3], color='green')
    # plt.scatter(-cam_poses[1, 1, 3], cam_poses[1, 2, 3], color='red')
    # plt.scatter(-poses_velo_w[selected_frames[0], 1, 3], poses_velo_w[selected_frames[0], 2, 3], color='black')
    # plt.scatter(-poses_imu_w[selected_frames[0], 1, 3], poses_imu_w[selected_frames[0], 2, 3], color='orange')
    # plt.axis('equal')


    # x_plt = 0
    # y_plt = 2
    # plt.scatter(poses[:, x_plt, 3], poses[:, y_plt, 3], color='cyan')
    # for plt_no in range(poses.shape[0]):
    #     plt.arrow(poses[plt_no, x_plt, 3], poses[plt_no, y_plt, 3], poses[plt_no, x_plt, 0], poses[plt_no, y_plt, 0],
    #               color='black')
    #     plt.arrow(poses[plt_no, x_plt, 3], poses[plt_no, y_plt, 3], poses[plt_no, x_plt, 1], poses[plt_no, y_plt, 1])
    #     plt.arrow(poses[plt_no, x_plt, 3], poses[plt_no, y_plt, 3], poses[plt_no, x_plt, 2], poses[plt_no, y_plt, 2],
    #               color='blue')

    # visible_objects = visible_objects_tracking
    # objects_meta = objects_meta_tracking

    images = np.concatenate(images_ls)
    poses = np.concatenate(poses_ls)

    objects_meta = objects_meta_ls[0]
    N_obj = np.array([len(seq_objs[0]) for seq_objs in visible_objects_ls]).max()
    for seq_i, visible_objects in enumerate(visible_objects_ls):
        diff = N_obj - len(visible_objects[0])
        if diff > 0:
            fill = np.ones([np.shape(visible_objects)[0], diff, np.shape(visible_objects)[2]]) * -1
            visible_objects = np.concatenate([visible_objects, fill], axis=1)
            visible_objects_ls[seq_i] = visible_objects

        if seq_i != 0:
            objects_meta.update(objects_meta_ls[seq_i])

    visible_objects = np.concatenate(visible_objects_ls)

    H = images.shape[1]
    W = images.shape[2]

    bboxes = None

    count = np.array(range(len(images)))
    i_split = [np.sort(count[:]),
               count[int(0.8 * len(count)):],
               count[int(0.8 * len(count)):]]

    novel_view = 'left'
    shift_frame = None
    n_oneside = int(poses.shape[0] / 2)
    render_poses = poses[:1]
    # Novel view middle between both cameras:
    if novel_view == 'mid':
        new_poses_o = ((poses[n_oneside:, :, -1] - poses[:n_oneside, :, -1]) / 2) + poses[:n_oneside, :, -1]
        new_poses = np.concatenate([poses[:n_oneside, :, :-1], new_poses_o[...,None]], axis=2)
        render_poses = new_poses

    elif novel_view == 'shift':
        render_poses = np.repeat(np.eye(4)[None], n_oneside, axis=0)
        l_poses = poses[:n_oneside, ...]
        r_poses = poses[n_oneside:, ...]
        render_poses[:, :3, :3] = (l_poses[:, :3, :3] + r_poses[:, :3, :3]) / 2.
        render_poses[:, :3, 3] = l_poses[:, :3, 3] + \
                                 (r_poses[:, :3, 3] - l_poses[:, :3, 3]) * np.linspace(0, 1, n_oneside)[:, None]
        if shift_frame is not None:
            visible_objects = np.repeat(visible_objects[shift_frame][None], len(visible_objects), axis=0)

    elif novel_view == 'left':
        render_poses = None
        start_i = 0
        # Render at trained left camera pose
        for seq_i, sequ in enumerate(selected_frames[0]):
            sequ_frames = [sequ, selected_frames[1][seq_i]]
            l_sequ = sequ_frames[1] - sequ_frames[0] + 1
            render_poses = poses[start_i:start_i+l_sequ, ...] if render_poses is None \
                else np.concatenate([render_poses, poses[start_i:start_i+l_sequ, ...]])
            start_i += 2*l_sequ
    elif novel_view == 'right':
        # Render at trained left camera pose
        render_poses = poses[n_oneside:, ...]

    render_objects = None
    if use_obj:
        start_i = 0
        # Render at trained left camera pose
        for seq_i, sequ in enumerate(selected_frames[0]):
            sequ_frames = [sequ, selected_frames[1][seq_i]]
            l_sequ = sequ_frames[1] - sequ_frames[0] + 1
            render_objects = visible_objects[start_i:start_i+l_sequ, ...] if render_objects is None \
                else np.concatenate([render_objects, visible_objects[start_i:start_i+l_sequ, ...]])
            start_i += 2*l_sequ

    if use_time:
        time_stamp = np.zeros([len(poses), 3])
        print('TIME ONLY WORKS FOR SINGLE SEQUENCES')
        time_stamp[:, 0] = np.repeat(np.linspace(selected_frames[0], selected_frames[1], len(poses) // 2)[None], 2,
                                     axis=0).flatten()
        render_time_stamp = time_stamp
    else:
        time_stamp = None
        render_time_stamp = None

    for i_seq, seq_start in enumerate(selected_frames[0]):
        seq_end = selected_frames[0][i_seq]
        if remove >= 0:
            print('REMOVE ONLY WORKS FOR SINGLE SEQUENCES')
            if remove > seq_end or remove < seq_start:
                print('No frame will be removed from the training sequence, since ', remove,' is not in the sequence')
            else:
                id_remove = remove - seq_start
                i_split[0] = np.delete(i_split[0], [id_remove, id_remove + len(poses) // 2])
                # time_stamp = np.delete(time_stamp, [id_remove, id_remove+len(poses)//2], axis=0) if time_stamp is not None else None
                # poses = np.delete(poses, [id_remove, id_remove+len(poses)//2], axis=0)
                # images = np.delete(images, [id_remove, id_remove+len(poses)//2], axis=0)

    return images, poses, render_poses, [H, W, focal], i_split, visible_objects, objects_meta, render_objects, bboxes, kitti_obj_metadata, time_stamp, render_time_stamp


# KITTI
# [frame_no,
# tracking_id,
# obj_type,
# truncated,
# occluded,
# alpha,
# bbox_l, bbox_t, bbox_r, bbox_b,
# dim_h, dim_w, dim_l,
# x, y, z,
# rot_y,
# (score)]

centertrack_map = {'tracking_id': 'tracking_id',
                   'obj_type': 'class',
                   'truncated': 0,
                   'occluded': 0,
                   'alpha': 'alpha',
                   'dimension': 'dim',
                   'location': 'loc',
                   'rot_y': 'rot_y'
                   }

# KITTI
#'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'

centertrack_type_map = {1: 'Car',
                        2: 'Truck',
                        3: 'Truck', #'bus',
                        4: 'Truck', # 'trailer': 4,
                        5: 'Truck',#'construction_vehicle': 5,
                        6: 'Pedestrian',
                        7: 'Cyclist', #'motorcycle': 7,
                        8:'Cyclist', #'bicycle': 8,
                        9: 'DontCare', #'traffic_cone': 9,
                        10: 'DontCare', #'barrier': 10
                        }


def tracking2txt(file):
    """
    Helper to convert json outputs of tracking algorithms to text files for scene graphs
    :param file: Json File
    :return:
    """
    # tst = True
    labels = []
    kitti2tracking = centertrack_map
    tracking2kitticlass = centertrack_type_map

    with open(file) as json_file:
        data = json.load(json_file)
        for i, frame_data in enumerate(data.values()):
            frame_objs = []
            for obj_data in frame_data:
                obj = [i]
                for kitti_val, tracked_val in kitti2tracking.items():
                    if not kitti_val == 'obj_type':
                        if isinstance(tracked_val, str):
                            obj.append(obj_data[tracked_val])
                        else:
                            obj.append(tracked_val)
                    else:
                        obj.append(tracking2kitticlass[obj_data[tracked_val]])

                frame_objs.append(obj)

            labels.append(frame_objs)

    with open('test_file.txt', 'w') as file:
        for frame in labels:
            for obj in frame:
                for attr in obj:
                    if isinstance(attr, list) > 1:
                        for sub_attr in attr:
                            file.write(str(sub_attr) + " ")
                    else:
                        file.write(str(attr)+ " ")
                file.write('\n')

        file.close()


def plot_kitti_poses(args, poses, visible_objects):
    """
    Plotting helper for scene graph poses (position + orientation) in global coordinates
    :param args:
    :param poses:
    :param visible_objects:
    :return:
    """
    plot_poses=True
    plot_obj = True
    ax_birdseye = [0, -1]  # if args.dataset_type == 'vkitti' else [0,1]
    ax_zy = [-1, 1]  # if args.dataset_type == 'vkitti' else [1,2]
    ax_xy = [0, 1]  # if args.dataset_type == 'vkitti' else [0,2]
    if plot_poses:
        fig, ax_lst = plt.subplots(2, 2)
        position = []
        for pose in poses:
            position.append(pose[:3, -1])

        position_array = np.array(position).astype(np.float32)
        plt.sca(ax_lst[0, 0])
        plt.scatter(position_array[:, ax_birdseye[0]], position_array[:, ax_birdseye[1]], color='b')
        plt.sca(ax_lst[1, 0])
        plt.scatter(position_array[:, ax_xy[0]], position_array[:, ax_xy[1]], color='b')
        plt.sca(ax_lst[0, 1])
        plt.scatter(position_array[:, ax_zy[0]], position_array[:, ax_zy[1]], color='b')

        if plot_obj:
            if args.dataset_type == 'vkitti':
                object_positions = np.concatenate(visible_objects[:, :, 7:10], axis=0)
            elif args.dataset_type == 'kitti':
                object_positions = np.concatenate(visible_objects[:, :, 7:10], axis=0)

            if not object_positions.shape[1] == 0:
                object_positions = np.squeeze(object_positions[np.argwhere(object_positions[:, 0] != -1)])
                object_positions = object_positions[None, :] if len(object_positions.shape) == 1 else object_positions
                plt.sca(ax_lst[0, 0])
                plt.scatter(object_positions[:, ax_birdseye[0]], object_positions[:, ax_birdseye[1]], color='black')
                plt.sca(ax_lst[1, 0])
                plt.scatter(object_positions[:, ax_xy[0]], object_positions[:, ax_xy[1]], color='black')
                plt.sca(ax_lst[0, 1])
                plt.scatter(object_positions[:, ax_zy[0]], object_positions[:, ax_zy[1]], color='black')

                # Get locale coordinates of the very first object
                plt.sca(ax_lst[0, 0])
                headings = np.reshape(visible_objects_plt[..., 10], [-1, 1])
                t_o_w = np.reshape(visible_objects_plt[..., 7:10], [-1, 3])
                # theta_0 = visible_objects_plt[0, 0, 10]
                # r_ov_0 = np.array([[np.cos(theta_0), 0, np.sin(theta_0)], [0, 1, 0], [-np.sin(theta_0), 0, np.cos(theta_0)]])
                for i, yaw in enumerate(headings):
                    yaw = float(yaw)
                    r_ov = np.array(
                        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
                    t = t_o_w[i]
                    x_v = np.matmul(np.concatenate([r_ov, t[:, None]], axis=1), [5., 0., 0., 1.])
                    z_v = np.matmul(np.concatenate([r_ov, t[:, None]], axis=1), [0., 0., 5., 1.])
                    v_origin = np.matmul(np.concatenate([r_ov, t[:, None]], axis=1), [0., 0., 0., 1.])

                    plt.arrow(v_origin[0], v_origin[2], x_v[0] - v_origin[0], x_v[2] - v_origin[2],
                              color='black', width=0.1)
                    plt.arrow(v_origin[0], v_origin[2], z_v[0] - v_origin[0], z_v[2] - v_origin[2],
                              color='orange', width=0.1)

        # For waymo  x --> -z, y --> x, z --> y
        x_c_0 = np.matmul(poses[0, :, :], np.array([5., .0, .0, 1.]))[:3]
        y_c_0 = np.matmul(poses[0, :, :], np.array([.0, 5., .0, 1.]))[:3]
        z_c_0 = np.matmul(poses[0, :, :], np.array([.0, .0, 5., 1.]))[:3]
        coord_cam_0 = [x_c_0, y_c_0, z_c_0]
        c_origin_0 = poses[0, :3, 3]

        plt.sca(ax_lst[0, 0])
        plt.arrow(c_origin_0[ax_birdseye[0]], c_origin_0[ax_birdseye[1]],
                  coord_cam_0[ax_birdseye[0]][ax_birdseye[0]] - c_origin_0[ax_birdseye[0]],
                  coord_cam_0[ax_birdseye[0]][ax_birdseye[1]] - c_origin_0[ax_birdseye[1]],
                  color='red', width=0.1)
        plt.arrow(c_origin_0[ax_birdseye[0]], c_origin_0[ax_birdseye[1]],
                  coord_cam_0[ax_birdseye[1]][ax_birdseye[0]] - c_origin_0[ax_birdseye[0]],
                  coord_cam_0[ax_birdseye[1]][ax_birdseye[1]] - c_origin_0[ax_birdseye[1]],
                  color='green', width=0.1)
        plt.axis('equal')
        plt.sca(ax_lst[1, 0])
        plt.arrow(c_origin_0[ax_xy[0]], c_origin_0[ax_xy[1]],
                  coord_cam_0[ax_xy[0]][ax_xy[0]] - c_origin_0[ax_xy[0]],
                  coord_cam_0[ax_xy[0]][ax_xy[1]] - c_origin_0[ax_xy[1]],
                  color='red', width=0.1)
        plt.arrow(c_origin_0[ax_xy[0]], c_origin_0[ax_xy[1]],
                  coord_cam_0[ax_xy[1]][ax_xy[0]] - c_origin_0[ax_xy[0]],
                  coord_cam_0[ax_xy[1]][ax_xy[1]] - c_origin_0[ax_xy[1]],
                  color='green', width=0.1)
        plt.axis('equal')
        plt.sca(ax_lst[0, 1])
        plt.arrow(c_origin_0[ax_zy[0]], c_origin_0[ax_zy[1]],
                  coord_cam_0[ax_zy[0]][ax_zy[0]] - c_origin_0[ax_zy[0]],
                  coord_cam_0[ax_zy[0]][ax_zy[1]] - c_origin_0[ax_zy[1]],
                  color='red', width=0.1)
        plt.arrow(c_origin_0[ax_zy[0]], c_origin_0[ax_zy[1]],
                  coord_cam_0[ax_zy[1]][ax_zy[0]] - c_origin_0[ax_zy[0]],
                  coord_cam_0[ax_zy[1]][ax_zy[1]] - c_origin_0[ax_zy[1]],
                  color='green', width=0.1)
        plt.axis('equal')

        # Plot global coord axis
        plt.sca(ax_lst[0, 0])
        plt.arrow(0, 0, 5, 0, color='cyan', width=0.1)
        plt.arrow(0, 0, 0, 5, color='cyan', width=0.1)