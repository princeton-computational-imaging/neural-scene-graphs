import os
import numpy as np
import imageio
import tensorflow as tf
import random
from matplotlib import pyplot as plt
from PIL import Image

_sem2label = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
}

_sem2model = {
    'Sedan4Door': 0,
    'Hatchback': 1,
    'Hybrid': 2,
    'SUV': 3,
    'Firetruck_small_eu': 4,
    'MCP2_Ambulance_A': 5,
    'Ford_F600_CargoVan': 6,
    'Renault_Kangoo': 7,
    'MCP2_BusA_01': 8,
}

_sem2color = {
    'Red': 0,
    'Silver': 1,
    'Blue': 2,
    'Black': 3,
    'White': 4,
    'Brown': 5,
    'Grey': 6,
}

def _convert_to_float(val):
    try:
        v = float(val)
        return v
    except:
        if val == 'True':
            return 1
        elif val == 'False':
            return 0
        else:
            ValueError('Is neither float nor boolean: ' + val)


def _get_kitti_information(path):
     f = open(path, 'r')
     c = f.read()
     c = c.split("\n", 1)[1]
     return np.array([[_convert_to_float(j) for j in i.split(' ')] for i in c.splitlines()])


def _get_scene_objects(basedir):
    """

    Args:
        basedir:

    Returns:
        objct pose:
            rame cameraID trackID
            alpha width height length
            world_space_X world_space_Y world_space_Z
            rotation_world_space_y rotation_world_space_x rotation_world_space_z
            camera_space_X camera_space_Y camera_space_Z
            rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z
            is_moving
        vehicles_meta:
            trackID
            onehot encoded Label
            onehot encoded vehicle model
            onehot encoded color
            3D bbox dimension (length, height, width)
        max_obj:
            Maximum number of objects in a single frame
        bboxes_by_frame:
            2D bboxes
    """
    object_pose = _get_kitti_information(os.path.join(basedir, 'pose.txt'))
    print('Loading poses from: ' + os.path.join(basedir, 'pose.txt'))
    bbox = _get_kitti_information(os.path.join(basedir, 'bbox.txt'))
    print('Loading bbox from: ' + os.path.join(basedir, 'bbox.txt'))
    info = open(os.path.join(basedir, 'info.txt')).read()
    print('Loading info from: ' + os.path.join(basedir, 'info.txt'))
    info = info.splitlines()[1:]

    # Creates a dictionary which label and model for each track_id
    vehicles_meta = {}

    for i, vehicle in enumerate(info):
        # Vehicle
        # label = np.zeros([len(_sem2label)])
        # model = np.zeros([len(_sem2model)])
        # color = np.zeros([len(_sem2color)])
        vehicle = vehicle.split()  # Ignores colour for now

        # label[_sem2label[vehicle[1]]] = 1
        # model[_sem2model[vehicle[2]]] = 1
        # color[_sem2color[vehicle[3]]] = 1

        label = np.array([_sem2label[vehicle[1]]])

        track_id = np.array([int(vehicle[0])])

        # width height length
        vehicle_dim = object_pose[np.where(object_pose[:, 2] == track_id), :][0, 0, 4:7]
        # For vkitti2 dimensions are defined: width height length
        # To Match vehicle axis xyz swap to length, height, width
        vehicle_dim = vehicle_dim[[2, 1, 0]]

        # vehicle = np.concatenate((np.concatenate((np.concatenate((track_id, label)), model)), color))
        vehicle = np.concatenate([track_id, vehicle_dim])
        vehicle = np.concatenate([vehicle, label])
        vehicles_meta[int(track_id)] = vehicle

    # Get the maximum number of objects in a single frame to define networks
    # input size for the specific scene if objects are used
    max_obj = 0
    f = 0
    c = 0
    count = 0
    for obj in object_pose[:,:2]:
        count += 1
        if not obj[0] == f or obj[1] == c:
            f = obj[0]
            c = obj[1]
            if count > max_obj:
                max_obj = count
            count = 0

    # Add to object_pose if the object is moving between the current and the next frame
    # TODO: Use if moving information to decide if an Object is static or dynamic across the whole scene!!
    object_pose = np.column_stack((object_pose, bbox[:, -1]))

    # Store 2D bounding boxes of frames
    bboxes_by_frame = []
    last_frame = bbox[-1, 0].astype(np.int32)
    for cam in range(2):
        for i in range(last_frame + 1):
            bbox_at_i = np.squeeze(bbox[np.argwhere(bbox[:, 0] == i), :7])
            bboxes_by_frame.append(bbox_at_i[np.argwhere(bbox_at_i[:, 1] == cam), 3:7])


    return object_pose, vehicles_meta, max_obj, bboxes_by_frame


def _get_objects_by_frame(object_pose, object_meta, max_obj, n_cam, selected_frames, row_id):
    """

    Args:
        object_pose: dynamic information in world and camera space for each object sorted by frames
        object_meta: metadata descirbing object properties like model, label, color, dimensions
        max_obj: Maximum number of objects in a single frame for the whole scene
        n_cam: Amount of cameras
        selected_frames: [first_frame, last_frame]
        row_id: bool

    Returns:
        visible_objects: all objects in the selected sequence of frames
        max_obj: Maximum number of objects in the selected sequence of frames
    """
    visible_objects = []
    frame_objects = []
    max_in_frames = 0
    const_pad = (0, 0)

    #### DEBUG
    ### TODO: Later specify ignored objects as arg
    ignore_objs = [16., 17., 18., 19.] # [12.]

    if row_id:
        const_pad = (-1, -1)

    for cam in range(n_cam):
        for obj_pose in object_pose:
            if obj_pose[2] not in ignore_objs:
                if obj_pose[1] == cam:
                    if selected_frames[0] <= obj_pose[0] <= selected_frames[1]:
                        if frame_objects:
                            if not all(frame_objects[-1][:2] == obj_pose[:2]):
                                max_in_frames = len(frame_objects) if max_in_frames < len(frame_objects) else max_in_frames
                                frame_objects = np.pad(np.array(frame_objects),
                                                       ((0, max_obj - len(frame_objects)), (0, 0)),
                                                       'constant', constant_values=const_pad)
                                visible_objects.append(frame_objects)
                                frame_objects = []

                        label = object_meta[obj_pose[2]][1:4]
                        obj_pose = np.concatenate((obj_pose, label))

                        frame_objects.append(obj_pose)

    max_in_frames = len(frame_objects) if max_in_frames < len(frame_objects) else max_in_frames
    frame_objects = np.pad(np.array(frame_objects),
                           ((0, max_obj - len(frame_objects)), (0, 0)),
                           'constant', constant_values=const_pad)
    visible_objects.append(frame_objects)

    if max_in_frames < max_obj:
        max_obj = max_in_frames

    # Remove all non existent objects from meta:
    object_meta_seq = {}
    for track_id in np.unique(np.array(visible_objects)[:, :, 2]):
        if track_id in object_meta:
            object_meta_seq[track_id] = object_meta[track_id]



    return visible_objects, object_meta_seq, max_obj


def load_vkitti_data(basedir, selected_frames=None, use_obj=True, row_id=False):
    """loads vkitti data

    Args:
        basedir: directory with frames, poses, extrinsic, ... as defined in vkitti2
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

    extrinsic = _get_kitti_information(os.path.join(basedir, 'extrinsic.txt'))
    intrinsic = _get_kitti_information(os.path.join(basedir, 'intrinsic.txt'))
    object_pose, object_meta, max_objects_per_frame, bboxes = _get_scene_objects(basedir)

    count = []
    imgs = []
    poses = []
    extrinsics = []
    frame_id = []
    instance_segm = []
    object_meta_seq = {}

    rgb_dir = os.path.join(basedir, 'frames/rgb')
    instance_dir = os.path.join(basedir, 'frames/instanceSegmentation')
    n_cam = len(os.listdir(rgb_dir))

    if selected_frames == -1:
        selected_frames = [0, extrinsic.shape[0]-1]

    for camera in next(os.walk(rgb_dir))[1]:
        frame_dir = os.path.join(rgb_dir, camera)
        instance_frame_dir = os.path.join(instance_dir, camera)
        cam = int(camera.split('Camera_')[1])

        # TODO: Check mismatching numbers of poses and Images like in loading script for llf
        for frame in sorted(os.listdir(frame_dir)):
            if frame.endswith('.jpg'):
                frame_num = int(frame.split('rgb_')[1].split('.jpg')[0])

                if selected_frames[0] <= frame_num <= selected_frames[1]:
                    fname = os.path.join(frame_dir, frame)
                    imgs.append(imageio.imread(fname))

                    inst_frame = 'instancegt_' + frame.split('rgb_')[1].split('.jpg')[0] + '.png'
                    instance_gt_name = os.path.join(instance_frame_dir, inst_frame)
                    if os.path.isdir(instance_gt_name):
                        instance_segm_img = Image.open(instance_gt_name)
                        instance_segm.append(np.reshape(np.array(instance_segm_img.getdata()), [imgs[0].shape[0], imgs[0].shape[1]]))

                    ext = extrinsic[frame_num * n_cam: frame_num * n_cam + n_cam, :][cam][2:]
                    ext = np.reshape(ext, (-1, 4))
                    extrinsics.append(ext)

                    # Get camera pose and location from extrinsics
                    pose = np.zeros([4, 4])
                    pose[3, 3] = 1
                    R = np.transpose(ext[:3, :3])
                    t = -ext[:3, -1]

                    # Camera position described in world coordinates
                    pose[:3, -1] = np.matmul(R, t)
                    # Match OpenGL definition of Z
                    pose[:3, :3] = np.matmul(np.eye(3), np.matmul(np.eye(3), R))
                    # Rotate pi around Z
                    pose[:3, 2] = -pose[:3, 2]
                    pose[:3, 1] = -pose[:3, 1]
                    poses.append(pose)
                    frame_id.append([frame_num, cam, 0])

                    count.append(len(imgs)-1)

    imgs = (np.array(imgs) / 255.).astype(np.float32)
    instance_segm = np.array(instance_segm)
    poses = np.array(poses).astype(np.float32)

    selected_range_cam0 = np.array(range(selected_frames[0], selected_frames[1]+1))
    selected_range_cam1 = np.array(range(len(bboxes)//2 + selected_frames[0], len(bboxes)//2 + selected_frames[1]+1))
    selected_range = np.concatenate([selected_range_cam0, selected_range_cam1])
    bboxes = np.array(bboxes)[selected_range]

    set_new_world = False
    if set_new_world:
        # Transform World System to first point in sequence
        pose_o = np.zeros([4, 4])
        pose_o[3, 3] = 1
        ext_0 = np.reshape(extrinsic[0, 2:], (-1, 4))
        pose_o[:3, -1] = -np.matmul(np.transpose(ext_0[:3, :3]), -ext_0[:3, 3])
        # Match OpenGL definition of Z
        pose_o[:3, :3] = np.matmul(np.array([[-1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, -1]]), np.transpose(ext_0[:3, :3])).T

        poses = np.matmul(pose_o[None, ...], poses)
        object_pose[:, 7:10] = np.matmul(pose_o[None, ...],
                                         np.concatenate([object_pose[:, 7:10],
                                                         np.ones([object_pose.shape[0], 1])], axis=1)[:, :, None])[:, :3, 0]

        object_rot_y = object_pose[10]

    visible_objects, object_meta, max_objects_per_frame = _get_objects_by_frame(object_pose,
                                                                                object_meta,
                                                                                    max_objects_per_frame,
                                                                                    n_cam,
                                                                                    selected_frames,
                                                                                    row_id)
    visible_objects = np.array(visible_objects).astype(np.float32)
    # TODO: Undo for final version, now speed up and overfit on less objects
    visible_objects = visible_objects[:, :max_objects_per_frame, :]

    if not use_obj:
        max_objects_per_frame = 0
        render_objects = None

    random.shuffle(count)

    H, W = imgs[0].shape[:2]
    focal = intrinsic[0, 2]

    i_split = [np.sort(count[:]),
               count[int(0.8 * len(count)):],
               count[int(0.8 * len(count)):]]

    novel_view = 'left'
    n_oneside = int(poses.shape[0]/2)

    render_poses = poses[:1]
    # Novel view middle between both cameras:
    if novel_view == 'mid':
        new_poses_o = ((poses[n_oneside:, :, -1] - poses[:n_oneside, :, -1]) / 2) + poses[:n_oneside, :, -1]
        new_poses = np.concatenate([poses[:n_oneside, :, :-1], new_poses_o[...,None]], axis=2)
        render_poses = new_poses

    elif novel_view == 'left':
        # Render at trained left camera pose
        render_poses = poses[:n_oneside, ...]
    elif novel_view == 'right':
        # Render at trained left camera pose
        render_poses = poses[n_oneside:, ...]

    # render_objects
    if use_obj:
        render_objects = visible_objects[:n_oneside, ...]
        random_render_obj = True
        if random_render_obj:
            n_render = render_poses.shape[0]
            random_obj_id = count
            random.shuffle(random_obj_id)
            render_objects = visible_objects[random_obj_id[:n_render], ...]
    else:
        render_objects = None

    bboxes = None

    return imgs, instance_segm, poses, frame_id, render_poses, [H, W, focal], i_split, visible_objects, object_meta, render_objects, bboxes