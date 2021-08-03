import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def spin_object(obj_pose, steps=11, min=-np.pi/30, max=np.pi/30):
    obj_pose = np.reshape(obj_pose, [-1, 6])

    rotation_poses = np.repeat(obj_pose, steps, axis=0)

    rotation_poses[:, 3] += np.linspace(min, max, steps)

    return np.reshape(rotation_poses, [steps, -1, 3])


def dancing_objects(obj_i, steps=30, min=-np.pi*1/5, max=np.pi*1/5):
    obj_dancing = -1 * np.ones([steps, obj_i.shape[0], obj_i.shape[1]])
    for k in range(obj_i.shape[0]//2):
        obj_to_spin = obj_i[2 * k:2 * (k + 1)]
        if obj_to_spin[0, 0] != -1:
            spined_obj = spin_object(obj_to_spin, steps, min, max)
            obj_dancing[:, 2 * k:2 * (k + 1), :] = spined_obj

    return obj_dancing

def move_along_track(objs):

    return objs

def ghost_street(objs):
    selected_obj = np.random.choice

    return objs


def remove_obj_from_set(obj_set, obj_meta, rm_obj):
    rm_id = np.where(np.equal(obj_meta[:, 0], rm_obj))[0]
    rm_lines = np.where(obj_set[:, :, 1] == rm_id)
    obj_set[rm_lines] = np.array([-1., 0., 0.])
    obj_set[rm_lines[0], rm_lines[1] - 1] = np.array([-1., 1., -1.])

    return obj_set



def manipulate_obj_pose(manipulation, obj, obj_meta, i, rm_obj=None):
    ''' composing scenes via neural scene graph manipulation functions such as:
            - rotate
            - dance
            - move
            - switch_location
            - oversample
            - traffic_jam
            - handcrafted_set_xx

    Args:
        manipulation: requested manipulation
        obj: scene graph nodes and edges at all times in the sequence
        obj_meta: object metadata (latent codes, size)
        i: frame in sequence
        rm_obj: removable objects
    Returns:
        render_set
    '''
    render_set = []
    if rm_obj is not None:
        if isinstance(rm_obj, list):
            for obj_instance in rm_obj:
                obj = remove_obj_from_set(obj, obj_meta, obj_instance)
        else:
            obj = remove_obj_from_set(obj, obj_meta, rm_obj)

    if manipulation is None:
        obj_i = tf.cast(obj[i], tf.float32)
        obj_i = tf.reshape(obj_i, [obj_i.shape[0] // 2, 2 * 3])
        render_set.append([obj_i])

    elif manipulation == 'rotate':
        select_obj = np.random.choice(np.linspace(0, len(obj[0])//2 - 1,len(obj[0])//2).astype(np.int32))
        rotate_obj = spin_object(obj[i, select_obj*2:(select_obj*2+2)])
        obj = np.repeat(obj[i][None], rotate_obj.shape[0], axis=0)
        obj = np.ones_like(obj) * -1.
        obj[..., 0:2, :] = rotate_obj

        for obj_i in obj:
            obj_i = tf.cast(obj_i, tf.float32)
            obj_i = tf.reshape(obj_i, [obj_i.shape[0] // 2, 2 * 3])
            render_set.append([obj_i])

    elif manipulation == 'dance':
        obj = dancing_objects(obj[i])

        for obj_i in obj:
            obj_i = tf.cast(obj_i, tf.float32)
            obj_i = tf.reshape(obj_i, [obj_i.shape[0] // 2, 2 * 3])
            render_set.append([obj_i])

    elif manipulation == 'move':
        obj_flat = np.reshape(obj, [-1, 6])
        ids = np.unique(obj_flat[:, 4])
        rand_moving_id = np.random.choice(ids)

        render_obj_poses = obj_flat[np.where(obj_flat[..., 4] == rand_moving_id)]

        obj = -1 * np.ones([render_obj_poses.shape[0], obj.shape[1]//2, 6])
        obj[..., 4] = 0.
        obj[:, 0, :] = render_obj_poses

        for obj_i in obj:
            obj_i = tf.cast(obj_i, tf.float32)
            render_set.append([obj_i])

    elif manipulation == 'switch_location':
        obj_i = np.reshape(obj[i], [-1, 6])
        obj_ids = obj_i[:, 4]
        np.random.shuffle(obj_ids)
        obj_i[:, 4] = obj_ids

        obj_i = tf.cast(obj[i], tf.float32)
        obj_i = tf.reshape(obj_i, [obj_i.shape[0] // 2, 2 * 3])
        render_set.append([obj_i])

    elif manipulation == 'oversample':
        factor = 2
        N_obj = len(obj[0]) // 2 * factor
        obj_flat = np.reshape(obj, [-1, 6])
        avg_z = np.mean(obj_flat[:, 1])
        std_z = np.std(obj_flat[:, 1])
        avg_angle = np.mean(obj_flat[:, 3]) + 0.04
        std_angle = np.std(obj_flat[:, 3])
        max_x = np.abs(obj_flat[:, 0]).max()
        max_y = 20 #  np.abs(obj_flat[:, 2]).max()

        thresh = np.mean(obj_meta[1:, 1]) * 1.2

        obj_pose_flat = []
        obj_flat = []

        for i in range(N_obj):
            close = True
            j = 0
            while close or j > 5:
                obj_pose = np.random.uniform(np.array([0, -std_z, -max_y, -std_angle]),
                                             np.array([max_x, std_z, max_y, std_angle])) \
                              + np.array([0., avg_z, 0., avg_angle])

                if obj_pose_flat:
                    min_distance = np.linalg.norm(np.array(obj_pose_flat)[:, :3] - obj_pose[:3], axis=1).min()
                    if min_distance > thresh:
                        obj_pose_flat.append(obj_pose)
                        close = False
                else:
                    obj_pose_flat.append(obj_pose)
                    close = False

            obj_id = np.random.choice(np.array(obj_meta)[1:, 0])
            indexing = np.argwhere(np.array(obj_meta)[:, 0] == obj_id)[0,0]
            obj_classe = obj_meta[indexing, 4]
            obj_i = np.concatenate([obj_pose, indexing[None], obj_classe[None]], axis=0)
            obj_flat.append(obj_i)

        obj_i = tf.cast(np.array(obj_flat), tf.float32)
        render_set.append([obj_i])

    elif manipulation == 'traffic_jam':
        obj_flat = np.reshape(obj, [-1, 6])
        obj_ids = np.unique(obj_flat[:, 4])[1:]
        np.random.shuffle(obj_flat)
        i = 0

        while i < len(obj_flat):
            obj_i_pose = obj_flat[i, [0, 2]]

            # id_i = int(np.random.choice(obj_ids))
            # obj_flat[i, 4] = id_i

            id_i = int(obj_flat[i, 4])
            thresh_close = np.array(id_i).max() * 1.5
            close_poses = []
            i += 1
            for k, pose_k in enumerate(obj_flat[i:]):
                x_k, y_k = pose_k[[0, 2]]
                d_i_k = np.sqrt((x_k - obj_i_pose[0])**2
                                + (y_k - obj_i_pose[1])**2)

                if d_i_k < thresh_close:
                    close_poses.append(i+k)

            obj_flat = np.delete(obj_flat, np.array(close_poses), axis=0)

        obj_flat = np.delete(obj_flat, tf.where(obj_flat[:, 0] == -1), axis=0)
        obj_i = tf.cast(obj_flat, tf.float32)
        render_set.append([obj_i])

    elif manipulation == 'background':
        obj_i = np.reshape(obj[i], [-1, 6])
        obj_i = -1 * np.ones_like(obj_i)
        obj_i[..., 4] = 0.
        obj_i = tf.cast(obj_i, tf.float32)
        render_set.append([obj_i])

    elif manipulation == 'translate':
        k = 15
        obj_old = np.reshape(obj[k], [-1, 6])
        print(obj_old)
        for j in range(len(obj_old)):
            for d in np.linspace(-2., 2., 15):
                obj_i = -1 * np.ones_like(obj_old)
                obj_i[0] = obj_old[j] + np.array([0., 0., d, 0., 0., 0.])
                obj_i = tf.cast(obj_i, tf.float32)
                render_set.append([obj_i])

    return render_set


