import time
import os
import numpy as np
import tensorflow as tf
from neural_scene_graph_helper import box_pts


def extract_object_information(args, visible_objects, objects_meta):
    '''Get object and object network properties for the given sequence

    Args:
        args:
            args.object_setting are experimental settings for object networks inputs, set to 0 for current version
        visible_objects: Objects per frame + Pose and other dynamic properties + tracking ID
        objects_meta: Metadata with additional static object information sorted by tracking ID

    Retruns:
        obj_properties [n_input_frames, n_max_objects, n_object_properties]: Object properties per frame
        add_input_rows: additional input rows to the network
    '''
    if args.dataset_type == 'vkitti':
        # [n_frames, n_max_obj, xyz+track_id+ismoving+0]
        obj_state = visible_objects[:, :, [7, 8, 9, 2, -1]]

        obj_dir = visible_objects[:, :, 10][..., None]
        # [..., width+height+length]
        # obj_dim = visible_objects[:, :, 4:7]
        sh = obj_state.shape

    elif args.dataset_type == 'waymo_od':
        obj_state = visible_objects[:, :, [7, 8, 9, 2, -1]]
        obj_dir = visible_objects[:, :, 10][..., None]
        sh = obj_state.shape
    elif args.dataset_type == 'kitti':
        obj_state = visible_objects[:, :, [7, 8, 9, 2, 3]]
        obj_dir = visible_objects[:, :, 10][..., None]
        sh = obj_state.shape

    # obj_state: [cam, n_obj, [x,y,z,track_id, class_id]]


    # [n_frames, n_max_obj]
    obj_track_id = obj_state[..., 3][..., None]
    obj_class_id = obj_state[..., 4][..., None]
    # Change track_id to row in list(objects_meta)
    obj_meta_ls = list(objects_meta.values())
    # Add first row for no objects
    obj_meta_ls.insert(0, np.zeros_like(obj_meta_ls[0]))
    obj_meta_ls[0][0] = -1
    # Build array describing the relation between metadata IDs and where its located
    row_to_track_id = np.concatenate([np.linspace(0, len(objects_meta.values()), len(objects_meta.values())+1)[:,None],
                                      np.array(obj_meta_ls)[:,0][:,None]], axis=1).astype(np.int32)
    # [n_frames, n_max_obj]
    track_row = np.zeros_like(obj_track_id)

    scene_objects = []
    scene_classes = list(np.unique(np.array(obj_meta_ls)[..., 4]))
    for i, frame_objects in enumerate(obj_track_id):
        for j, camera_objects in enumerate(frame_objects):
            track_row[i, j] = np.argwhere(row_to_track_id[:, 1] == camera_objects)
            if camera_objects >= 0 and not camera_objects in scene_objects:
                print(camera_objects, 'in this scene')
                scene_objects.append(camera_objects)

    obj_properties = np.concatenate([obj_state[..., :3], obj_dir, track_row], axis=2)

    if obj_properties.shape[-1] % 3 > 0:
        if obj_properties.shape[-1] % 3 == 1:
            obj_properties = np.concatenate([obj_properties, np.zeros([sh[0], sh[1], 2])], axis=2)
        else:
            obj_properties = np.concatenate([obj_properties, np.zeros([sh[0], sh[1], 1])], axis=2)

    add_input_rows = int(obj_properties.shape[-1] / 3)

    obj_meta_ls = [obj * np.array([1., args.box_scale, 1., args.box_scale, 1.])
                   if obj[4] != 4 else obj * np.array([1., 1.2, 1., 1.2, 1.])
                   for obj in obj_meta_ls]
    # obj_meta_ls = [obj * np.array([1., args.box_scale, 1., args.box_scale, 1.]) for obj in obj_meta_ls]

    return obj_properties, add_input_rows, obj_meta_ls, scene_objects, scene_classes


def plane_bounds(poses, plane_type, near, far, N_samples):
    ''' Define Plane bounds and plane index array

    Args:
        poses: camera poses
        plane_type: selects the specific distribution of samples
        near: closest sampling point along a ray
        far: minimum distance to last plane in the scene
        N_samples: amount of steps along each ray

    Returns:
        plane_bds: first and last sampling plane in the scene
        plane_normal: plane normals
        plane_delta: distance between each plane
        id_planes: id of planes selected for sampling give a specific plane_type
        near: distance to the closest samping point along a ray
        far: distance to last plane in the scene
    '''

    # [N_poses*N_samples, xyz]
    plane_normal = -poses[0, :3, 2]

    # The first plane in front of the first pose in the scene [N_poses, N_samples, xyz]
    first_plane = poses[0, :3, -1] + near * plane_normal

    # Current Assumption the vehicle is driving a straight line
    # For 2 cameras each half of the poses are similar
    n_left = int(poses.shape[0] / 2)

    # Distance between the first and last pose
    max_pose_dist = np.linalg.norm(poses[-1, :3, -1] - poses[0, :3, -1])

    # Distances between two frames
    if not n_left > 1:
        pose_dist = max_pose_dist + 1e-9
    else:
        pose_dist = np.linalg.norm(poses[0:n_left - 1, :3, -1] - poses[1:n_left, :3, -1], axis=1)

    if plane_type == 'uniform':
        # Ensure in fornt of any point are equaly or more planes than Sample+Importnace
        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)

        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    if plane_type == 'uniform_exp':
        # The first plane in front of the first pose in the scene [N_poses, N_samples, xyz]
        first_plane = poses[0, :3, -1] + near * 1.2 * plane_normal

        # Ensure in fornt of any point are equaly or more planes than Sample+Importnace
        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)

        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    elif plane_type == 'experimental':
        t = np.zeros([N_samples])
        id_planes = np.zeros([N_samples])
        t[1] = 1
        for i in range(N_samples - 1):
            t[i + 1] += t[i] * 1.7
            id_planes[i] = np.sum(t[:i + 1])

        id_planes[-1] = np.sum(t)
        plane_delta = (far - near) / id_planes[-1]
        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'double':
        t = np.zeros([N_samples])
        id_planes = np.zeros([N_samples])
        t[1] = 1
        for i in range(N_samples - 1):
            t[i + 1] += t[i] * 2
            id_planes[i] = np.sum(t[:i + 1])

        id_planes[-1] = np.sum(t)
        plane_delta = (far - near) / id_planes[-1]
        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'bckg' or plane_type == 'reversed':
        t = np.zeros([N_samples])
        id_planes = np.zeros([N_samples])
        t[1] = 1
        for i in range(N_samples - 1):
            t[i + 1] += t[i] * 1.
            id_planes[i] = np.ceil(np.sum(t[:i + 1]))

        id_planes[-1] = np.ceil(np.sum(t))
        id_planes = np.sort((id_planes[-1]-id_planes))
        plane_delta = (far - near) / id_planes[-1]

        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'non-uniform':
        # Adds depth+1*delta between each plane
        t = np.linspace(0, N_samples - 1, N_samples)
        id_planes = np.zeros([N_samples])
        for i in range(N_samples):
            id_planes[i] = np.sum(t[:i + 1])

        plane_delta = (far - near) / (id_planes[-1])
        add_planes = np.ceil(pose_dist.max() * n_left / plane_delta)

    elif plane_type == 'strict_uniform':
        first_plane = poses[-1, :3, -1] + near * plane_normal
        # Ensure in fornt of any point are equaly or more planes than Sample+Importnace
        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)

        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    elif plane_type == 'move':
        aprox_near_planes = round(max_pose_dist / near)
        aprox_delta = (far-near) / (N_samples-1)
        no_near_spaces = int(max_pose_dist / aprox_delta)+1
        if no_near_spaces > 1.:
            print('Selected planes might not work')

        plane_delta = near
        planes_per_section = np.ceil(aprox_delta / near)
        id_planes = np.linspace(0, N_samples - 1, N_samples) * planes_per_section

        add_planes = (no_near_spaces-1) * planes_per_section

    elif plane_type == 'static_move':
        first_plane = poses[n_left-1, :3, -1] + near * 1.2 * plane_normal

        id_planes = np.linspace(0, N_samples - 1, N_samples)
        plane_delta = (far - near) / (N_samples - 1)
        poses_per_plane = int(((far - near) / N_samples) / pose_dist.max())
        add_planes = np.ceil(n_left / poses_per_plane)

    last_plane = first_plane + ((id_planes[-1] + add_planes) * plane_delta) * plane_normal
    far = near + plane_delta * (id_planes[-1] + add_planes)
    plane_bds = np.concatenate([first_plane[:, None], last_plane[:, None]], axis=1)

    return plane_bds, plane_normal, plane_delta, id_planes, near, far


def get_bbox_pixel(bboxes, i_train, hwf):
    """get all rays hitting an ojects given a 2D object detection result

    Args:
        bboxes: 2D bounding boxes
        i_train: train split
        hwf: [Height, Width, focal length]

    Returns:
         rays_on_obj: All rays/gt pixels inside a 2D bounding box of an object
    """

    H, W, _ = hwf
    print('extract background')
    rays_on_obj = []
    pixel_offset = 2
    for i, bboxes_in_frames in enumerate(bboxes[i_train]):
        start_ray = i * H * W

        for box in bboxes_in_frames:
            l_b = np.squeeze(box[0][..., 0]).astype(np.int32) - pixel_offset
            r_b = np.squeeze(box[0][..., 1]).astype(np.int32) + pixel_offset
            top_b = np.squeeze(box[0][..., 2]).astype(np.int32) - pixel_offset
            bot_b = np.squeeze(box[0][..., 3]).astype(np.int32) + pixel_offset

            l_to_r = np.minimum(np.maximum(np.array(range(l_b, r_b + 1)), 0), W)[None, :]
            t_to_b = np.minimum(np.maximum(np.array(range(top_b, bot_b + 1)), 0), H)[:, None]

            bbox_pixels = t_to_b * W + np.repeat(l_to_r, t_to_b.shape[0], axis=0) + start_ray
            bbox_pixels = bbox_pixels.flatten('C')
            rays_on_obj.append(bbox_pixels)

    rays_on_obj = np.array(np.concatenate(rays_on_obj))
    rays_on_obj = np.delete(rays_on_obj, np.where(rays_on_obj > H * W * len(i_train) - 1)[0])

    return rays_on_obj


def get_all_ray_3dbox_intersection(rays_rgb, obj_meta_tensor, chunk, local=False, obj_to_remove=-100):
    '''get all rays hitting an oject given 3D multi-object-tracking results of a sequence

    Args:
        rays_rgb: All rays
        obj_meta_tensor: Metadata of all objects
        chunk: No. of rays processed at the same time
        local: Limit used memory if processed on a local machine with limited CPU/GPU resources
        obj_to_remove: If object should be removed from the set of rays

    Returns:
        rays_on_obj: Set of all rays hitting at least one object
        rays_to_remove: Set of all rays hitting an object, that should not be trained
    '''

    print('Removing object ', obj_to_remove)
    rays_on_obj = np.array([])
    rays_to_remove = np.array([])
    _batch_sz_inter = chunk if not local else 5000  # args.chunk
    _only_intersect_rays_rgb = rays_rgb[0][None]
    _n_rays = rays_rgb.shape[0]
    _n_obj = (rays_rgb.shape[1] - 3) // 2
    _n_bt = np.ceil(_n_rays / _batch_sz_inter).astype(np.int32)

    for i in range(_n_bt):
        _tf_rays_rgb = tf.cast(rays_rgb[i * _batch_sz_inter:(i + 1) * _batch_sz_inter], tf.float32)
        _n_bt_i = _tf_rays_rgb.shape[0]
        _rays_bt = [_tf_rays_rgb[:, 0, :], _tf_rays_rgb[:, 1, :]]
        _objs = tf.reshape(_tf_rays_rgb[:, 3:, :], [_n_bt_i, _n_obj, 6])
        _obj_pose = _objs[..., :3]
        _obj_theta = _objs[..., 3]
        _obj_id = tf.cast(_objs[..., 4], tf.int32)
        _obj_meta = tf.gather(obj_meta_tensor, _obj_id, axis=0)
        _obj_track_id = _obj_meta[..., 0, tf.newaxis]
        _obj_dim = _obj_meta[..., 1:4]

        _mask = box_pts(_rays_bt, _obj_pose, _obj_theta, _obj_dim, one_intersec_per_ray=False)[8]
        if _mask is not None:
            if rays_on_obj.any():
                rays_on_obj = np.concatenate([rays_on_obj, np.array(i * _batch_sz_inter + _mask[:, 0])])
            else:
                rays_on_obj = np.array(i * _batch_sz_inter + _mask[:, 0])
            if obj_to_remove is not None:
                _hit_id = tf.gather_nd(_obj_track_id, _mask)
                # bool_remove = tf.equal(_hit_id, obj_to_remove)
                bool_remove = np.equal(_hit_id, obj_to_remove)
                if any(bool_remove):
                    # _remove_mask = tf.gather_nd(_mask, tf.where(bool_remove))
                    _remove_mask = np.array(_mask[:, 0])[np.where(np.equal(_hit_id, obj_to_remove))[0]]
                    if rays_to_remove.any():
                        rays_to_remove = np.concatenate([rays_to_remove, np.array(i * _batch_sz_inter + _remove_mask)])
                    else:
                        rays_to_remove = np.array(i * _batch_sz_inter + _remove_mask)

    return rays_on_obj, rays_to_remove


def resample_rays(rays_rgb, rays_bckg, obj_meta_tensor, objects_meta, scene_objects, scene_classes, chunk, local=False):
    ''' Sample more rays for objects to even out classes and objects per batch

    Args:
        rays_rgb: All rays
        rays_bckg: Set of all rays hitting no object
        obj_meta_tensor: Metadata of all objects as tf array
        objects_meta: Metadata of all objects as np array
        scene_objects: Objects present in the viewed sequence
        scene_classes: Classes present in the viewed sequence
        chunk: No. of rays processed at the same time
        local: Limit used memory if processed on a local machine with limited CPU/GPU resources
    Returns:
        rays_rgb
    '''
    _batch_sz_inter = chunk if not local else 5000
    _n_rays = rays_rgb.shape[0]
    _n_obj = (rays_rgb.shape[1] - 3) // 2
    _n_bt = np.ceil(_n_rays / _batch_sz_inter).astype(np.int32)
    _obj_counts = np.zeros(np.max(np.array(scene_objects)).astype(np.int32) + 1)

    _new_rays_rgb = [None] * (np.max(np.array(scene_objects)).astype(np.int32) + 1)
    for i in range(_n_bt):
        _tf_rays_rgb = tf.cast(rays_rgb[i * chunk:(i + 1) * chunk], tf.float32)
        _n_bt_i = _tf_rays_rgb.shape[0]
        _rays_bt = [_tf_rays_rgb[:, 0, :], _tf_rays_rgb[:, 1, :]]
        _objs = tf.reshape(_tf_rays_rgb[:, 3:, :], [_n_bt_i, _n_obj, 6])
        _obj_pose = _objs[..., :3]
        _obj_theta = _objs[..., 3]
        _obj_id = tf.cast(_objs[..., 4], tf.int32)
        _obj_meta = tf.gather(obj_meta_tensor, _obj_id, axis=0)
        _obj_track_id = _obj_meta[..., 0, tf.newaxis]
        _obj_dim = _obj_meta[..., 1:4]

        _mask = box_pts(_rays_bt, _obj_pose, _obj_theta, _obj_dim, one_intersec_per_ray=True)[8]
        if _mask is not None:
            _hit_id = tf.gather_nd(_obj_track_id, _mask)

            for k in scene_objects:
                _k_mask = tf.gather(_mask, tf.where(tf.equal(_hit_id, k))[:, 0])
                _k_rays = tf.gather(_tf_rays_rgb, _k_mask[:, 0])

                _obj_counts[int(k)] += np.array(tf.reduce_sum(tf.cast(tf.equal(_hit_id, k), tf.int32)))
                if _new_rays_rgb[int(k)] is None:
                    _new_rays_rgb[int(k)] = []
                _new_rays_rgb[int(k)].append(_k_rays)

    # scene_classes = np.concatenate([np.array(scene_classes)[None], np.zeros(len(scene_classes))[None]])
    # for j, k in enumerate(scene_objects):
    #     _id_hit = int(k)
    #     obj_k_class = objects_meta[_id_hit][4]
    #     scene_objects[j] = np.stack([k[0], obj_k_class])
    #
    # for obj_class in scene_classes.T:
    #     for obj_k_c in np.where(np.array(scene_objects)[:, 1] == obj_class):
    #         obj_k_c
    #
    #
    #
    # obj_hits = np.zeros(np.array(scene_objects).max()+1)
    # for k in scene_objects:
    #     _id_hit = int(k)
    #     obj_k_class = objects_meta[_id_hit][4]
    #     class_id = np.where(scene_classes[0, :] == obj_k_class)
    #     print('Object', k, 'of class', obj_k_class, 'is hit by', _obj_counts[_id_hit])
    #     print(int(class_id))
    #     if _obj_counts[_id_hit] > 0:
    #         scene_classes[1, class_id] += _obj_counts[_id_hit]
    #         obj_hits[_id_hit] = _obj_counts[_id_hit]
    #
    # print(scene_classes)
    # print(obj_hits)

    # for obj_class in scene_classes.T:
    #     for k in scene_objects:
    #         _id_hit = int(k)
    #         obj_k_class = objects_meta[_id_hit][4]
    #         if obj_k_class == obj_class:

    rays_rgb = []

    unique_classes = np.unique(np.array(list(objects_meta.values()))[:, -1]).astype(np.int32)
    class_multiplier = dict.fromkeys(unique_classes)
    for i in unique_classes: class_multiplier[i] = 0

    for obj in list(objects_meta.values()):
        obj_class = obj[-1]
        class_multiplier[obj_class] += _obj_counts[int(obj[0])]

    hits_per_class = np.array(list(class_multiplier.values()))

    for key in class_multiplier:
        class_multiplier[key] = np.round((class_multiplier[key] / hits_per_class.max()) ** (-1))

    for k in scene_objects:
        _id_hit = int(k)
        if _obj_counts[_id_hit] > 0:
            _hit_factor = (np.max(_obj_counts) // _obj_counts[_id_hit]).astype(np.int32)
            print(_id_hit, 'is hit by', _obj_counts[_id_hit], 'rays!')
            print('This is', _hit_factor, 'times less than the most hit object!')

            # Manually add support for objects not present enough in specific datasets e.g. pedestrians in KITTI sequences
            if objects_meta[_id_hit][4] == 2 or objects_meta[_id_hit][4] == 1:
                _support_factor = class_multiplier[2]
                print('Adding Truck and Van support Factor', _support_factor)
                _hit_factor *= _support_factor
            if objects_meta[_id_hit][4] == 4:
                _support_factor = class_multiplier[4]
                print('Adding Pedestrian support Factor', _support_factor)
                _hit_factor *= _support_factor
            if objects_meta[_id_hit][4] == 0:
                _support_factor = class_multiplier[0]
                print('Adding Car support Factor', _support_factor)
                _hit_factor *= _support_factor


            _hit_factor = np.minimum(_hit_factor, 1e1)
            _eq_sz_rays = np.repeat(np.concatenate(np.array(_new_rays_rgb[_id_hit]), axis=0), _hit_factor, axis=0)
            rays_rgb.append(np.array(_eq_sz_rays))

    rays_rgb = np.concatenate(rays_rgb, axis=0)
    if rays_bckg is not None and not local:
        print('Adding dense sampling close to objects.')
        rays_rgb = np.concatenate([rays_rgb, rays_bckg], axis=0)
        print(rays_rgb.shape)

    del _new_rays_rgb
    del _objs
    del _tf_rays_rgb

    return rays_rgb
