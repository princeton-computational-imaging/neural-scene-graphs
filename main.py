import os
import random
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from neural_scene_graph_helper import *
from data_loader.load_vkitti import load_vkitti_data
from data_loader.load_kitti import load_kitti_data, plot_kitti_poses, tracking2txt
from prepare_input_helper import *
from neural_scene_graph_manipulation import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import cv2

tf.compat.v1.enable_eager_execution()


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, embedobj_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = tf.reshape(inputs[..., :3], [-1, 3])

    embedded = embed_fn(inputs_flat)
    if inputs.shape[-1] > 3:
        if inputs.shape[-1] == 4:
            # NeRF + T w/o embedding
            time_st = tf.reshape(inputs[..., 3], [inputs_flat.shape[0], -1])
            embedded = tf.concat([embedded, time_st], -1)
        else:
            # NeRF + Latent Code
            inputs_latent = tf.reshape(inputs[..., 3:], [inputs_flat.shape[0], -1])
            embedded = tf.concat([embedded, inputs_latent], -1)

    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None, :3], inputs[..., :3].shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

        if viewdirs.shape[-1] > 3:
            # Use global locations of objects
            input_obj_pose = tf.broadcast_to(viewdirs[:, None, 3:],
                                             shape=[inputs[..., :3].shape[0], inputs[..., :3].shape[1], 3])
            input_obj_pose_flat = tf.reshape(input_obj_pose, [-1, input_obj_pose.shape[-1]])
            embedded_obj = embedobj_fn(input_obj_pose_flat)
            embedded = tf.concat([embedded, embedded_obj], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                N_samples_obj,
                retraw=False,
                perturb=1.,
                N_importance=0,
                network_fine=None,
                object_network_fn_dict=None,
                latent_vector_dict=None,
                N_obj=None,
                obj_only=False,
                obj_transparency=True,
                white_bkgd=False,
                raw_noise_std=0.,
                sampling_method=None,
                use_time=False,
                plane_bds=None,
                plane_normal=None,
                delta=0.,
                id_planes=0,
                verbose=False,
                obj_location=True):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      object_network_fn_dict: dictinoary of functions. Model for predicting RGB and density at each point in
        object frames
      latent_vector_dict: Dictionary of latent codes
      N_obj: Maximumn amount of objects per ray
      obj_only: bool. If True, only run models from object_network_fn_dict
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      sampling_mehtod: string. Select how points are sampled in space
      plane_bds: array of shape [2, 3]. If sampling method planes, descirbing the first and last plane in space.
      plane_normal: array of shape [3]. Normal of all planes
      delta: float. Distance between adjacent planes.
      id_planes: array of shape [N_samples]. Preselected planes for sampling method planes and a given sampling distribution
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map


    def sample_along_ray(near, far, N_samples, N_rays, sampling_method, perturb):
        # Sample along each ray given one of the sampling methods. Under the logic, all rays will be sampled at
        # the same times.
        t_vals = tf.linspace(0., 1., N_samples)
        if sampling_method == 'squareddist':
            z_vals = near * (1. - np.square(t_vals)) + far * (np.square(t_vals))
        elif sampling_method == 'lindisp':
            # Sample linearly in inverse depth (disparity).
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        else:
            # Space integration times linearly between 'near' and 'far'. Same
            # integration points will be used for all rays.
            z_vals = near * (1.-t_vals) + far * (t_vals)
            if sampling_method == 'discrete':
                perturb = 0

        # Perturb sampling time along each ray. (vanilla NeRF option)
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([mids, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = tf.random.uniform(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        return tf.broadcast_to(z_vals, [N_rays, N_samples]), perturb


    ###############################
    # batch size
    N_rays = int(ray_batch.shape[0])

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    if use_time:
        time_stamp = ray_batch[:, 11][:, tf.newaxis]

    # Extract object position, dimension and label
    if N_obj:
        obj_pose = ray_batch[:, 11:]
        # [N_rays, N_obj, 8] with 3D position, y rot angle, track_id, (3D dimension - length, height, width)
        obj_pose = tf.reshape(obj_pose, [N_rays, N_obj, obj_pose.shape[-1] // N_obj])
        if N_importance > 0:
            obj_pose_fine = tf.repeat(obj_pose[:, tf.newaxis, ...], N_importance + N_samples, axis=1)
    else:
        obj_pose = obj_pose_fine = None

    if not obj_only:
        # For training object models only sampling close to the objects is performed
        if (sampling_method == 'planes' or sampling_method == 'planes_plus') and plane_bds is not None:
            # Sample at ray plane intersection (Neural Scene Graphs)
            pts, z_vals = plane_pts([rays_o, rays_d], [plane_bds, plane_normal, delta], id_planes, near,
                                    method=sampling_method)
            N_importance = 0
        else:
            # Sample along ray (vanilla NeRF)
            z_vals, perturb = sample_along_ray(near, far, N_samples, N_rays, sampling_method, perturb)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples, 3]

    ####### DEBUG Sampling Points
    # print('TURN OFF IF NOT DEBUGING!')
    # axes_ls = plt.figure(1).axes
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 0], np.array(rays_o)[i, 2],
    #               np.array(30 * rays_d)[i, 0],
    #               np.array(30 * rays_d)[i, 2], color='red')
    #
    # plt.sca(axes_ls[1])
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 2], np.array(rays_o)[i, 1],
    #               np.array(30 * rays_d)[i, 2],
    #               np.array(30 * rays_d)[i, 1], color='red')
    #
    # plt.sca(axes_ls[2])
    # for i in range(rays_o.shape[0]):
    #     plt.arrow(np.array(rays_o)[i, 0], np.array(rays_o)[i, 1],
    #               np.array(30 * rays_d)[i, 0],
    #               np.array(30 * rays_d)[i, 1], color='red')
    ####### DEBUG Sampling Points

    # Choose input options
    if not N_obj:
        # No objects
        if use_time:
            # Time parameter input
            time_stamp_fine = tf.repeat(time_stamp[:, tf.newaxis], N_importance + N_samples,
                                        axis=1) if N_importance > 0 else None
            time_stamp = tf.repeat(time_stamp[:, tf.newaxis], N_samples, axis=1)
            pts = tf.concat([pts, time_stamp], axis=-1)
            raw = network_query_fn(pts, viewdirs, network_fn)
        else:
            raw = network_query_fn(pts, viewdirs, network_fn)
    else:
        n_intersect = None
        if not obj_pose.shape[-1] > 5:
            # If no object dimension is given all points in the scene given in object coordinates will be used as an input to each object model
            pts_obj, viewdirs_obj = world2object(pts, viewdirs, obj_pose[..., :3], obj_pose[..., 3],
                                                 dim=obj_pose[..., 5:8] if obj_pose.shape[-1] > 5 else None)

            pts_obj = tf.transpose(tf.reshape(pts_obj, [N_rays, N_samples, N_obj, 3]), [0, 2, 1, 3])

            inputs = tf.concat([pts_obj, tf.repeat(obj_pose[..., None, :3], N_samples, axis=2)], axis=3)
        else:
            # If 3D bounding boxes are given get intersecting rays and intersection points in scaled object frames
            pts_box_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w,\
            pts_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
            intersection_map = box_pts(
                [rays_o, rays_d], obj_pose[..., :3], obj_pose[..., 3], dim=obj_pose[..., 5:8],
                one_intersec_per_ray=not obj_transparency)

            if z_vals_in_o is None or len(z_vals_in_o) == 0:
                if obj_only:
                    # No computation necesary if rays are not intersecting with any objects and no background is selected
                    raw = tf.zeros([N_rays, 1, 4])
                    z_vals = tf.zeros([N_rays, 1])

                    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                        raw, z_vals, rays_d)

                    rgb_map = tf.ones([N_rays, 3])
                    disp_map = tf.ones([N_rays])*1e10
                    acc_map = tf.zeros([N_rays])
                    depth_map = tf.zeros([N_rays])

                    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
                    if retraw:
                        ret['raw'] = raw
                    return ret
                else:
                    # TODO: Do not return anything for no intersections.
                    z_vals_obj_w = tf.zeros([1])
                    intersection_map = tf.cast(tf.zeros([1, 3]), tf.int32)

            else:
                n_intersect = z_vals_in_o.shape[0]

                obj_pose = tf.gather_nd(obj_pose, intersection_map)
                obj_pose = tf.repeat(obj_pose[:, tf.newaxis, :], N_samples_obj, axis=1)
                # Get additional model inputs for intersecting rays
                if N_samples_obj > 1:
                    z_vals_box_o = tf.repeat(tf.linspace(0., 1., N_samples_obj)[tf.newaxis, :], n_intersect, axis=0) * \
                                   (z_vals_out_o - z_vals_in_o)[:, tf.newaxis]
                else:
                    z_vals_box_o = tf.repeat(tf.constant(1/2)[tf.newaxis,tf.newaxis], n_intersect, axis=0) * \
                                   (z_vals_out_o - z_vals_in_o)[:, tf.newaxis]

                pts_box_samples_o = pts_box_o[:, tf.newaxis, :] + viewdirs_box_o[:, tf.newaxis, :] \
                                        * z_vals_box_o[..., tf.newaxis]
                # pts_box_samples_o = pts_box_samples_o[:, tf.newaxis, ...]
                # pts_box_samples_o = tf.reshape(pts_box_samples_o, [-1, 3])

                obj_pose_transform = tf.reshape(obj_pose, [-1, obj_pose.shape[-1]])

                pts_box_samples_w, _ = world2object(tf.reshape(pts_box_samples_o, [-1, 3]), None,
                                                    obj_pose_transform[..., :3],
                                                    obj_pose_transform[..., 3],
                                                    dim=obj_pose_transform[..., 5:8] if obj_pose.shape[-1] > 5 else None,
                                                    inverse=True)

                pts_box_samples_w = tf.reshape(pts_box_samples_w, [n_intersect, N_samples_obj, 3])

                z_vals_obj_w = tf.norm(pts_box_samples_w - tf.gather_nd(rays_o, intersection_map[:, 0, tf.newaxis])[:, tf.newaxis, :], axis=-1)

                # else:
                #     z_vals_obj_w = z_vals_in_w[:, tf.newaxis]
                #     pts_box_samples_o = pts_box_o[:, tf.newaxis, :]
                #     pts_box_samples_w = pts_box_w[:, tf.newaxis, :]

                #####
                # print('TURN OFF IF NOT DEBUGING!')
                # axes_ls = plt.figure(1).axes
                # plt.sca(axes_ls[0])
                #
                # pts = np.reshape(pts_box_samples_w, [-1, 3])
                # plt.scatter(pts[:, 0], pts[:, 2], color='red')
                ####

                # Extract objects
                obj_ids = obj_pose[..., 4]
                object_y, object_idx = tf.unique(tf.reshape(obj_pose[..., 4], [-1]))
                # Extract classes
                obj_class = obj_pose[..., 8]
                unique_classes = tf.unique(tf.reshape(obj_class, [-1]))
                class_id = tf.reshape(unique_classes.idx, obj_class.shape)

                inputs = pts_box_samples_o

                if latent_vector_dict is not None:
                    latent_vector_inputs = None

                    for y, obj_id in enumerate(object_y):
                        indices = tf.where(tf.equal(object_idx, y))
                        latent_vector = latent_vector_dict['latent_vector_obj_' + str(int(obj_id)).zfill(5)][tf.newaxis, :]
                        latent_vector = tf.repeat(latent_vector, indices.shape[0], axis=0)

                        latent_vector = tf.scatter_nd(indices, latent_vector, [n_intersect*N_samples_obj, latent_vector.shape[-1]])

                        if latent_vector_inputs is None:
                            latent_vector_inputs = latent_vector
                        else:
                            latent_vector_inputs += latent_vector

                    latent_vector_inputs = tf.reshape(latent_vector_inputs, [n_intersect, N_samples_obj, -1])
                    inputs = tf.concat([inputs, latent_vector_inputs], axis=2)

                # inputs = tf.concat([inputs, obj_pose[..., :3]], axis=-1)

                # objdirs = tf.concat([tf.cos(obj_pose[:, 0, 3, tf.newaxis]), tf.sin(obj_pose[:, 0, 3, tf.newaxis])], axis=1)
                # objdirs = objdirs / tf.reduce_sum(objdirs, axis=1)[:, tf.newaxis]
                # viewdirs_obj = tf.concat([viewdirs_box_o, obj_pose[..., :3][:, 0, :], objdirs], axis=1)
                if obj_location:
                    viewdirs_obj = tf.concat([viewdirs_box_o, obj_pose[..., :3][:, 0, :]], axis=1)
                else:
                    viewdirs_obj = viewdirs_box_o

        if not obj_only:
            # Get integration step for all models
            z_vals, id_z_vals_bckg, id_z_vals_obj = combine_z(z_vals,
                                                              z_vals_obj_w if z_vals_in_o is not None else None,
                                                              intersection_map,
                                                              N_rays,
                                                              N_samples,
                                                              N_obj,
                                                              N_samples_obj, )
        else:
            z_vals, _, id_z_vals_obj = combine_z(None, z_vals_obj_w, intersection_map, N_rays, N_samples, N_obj,
                                                 N_samples_obj)


        if not obj_only:
            # Run background model
            raw = tf.zeros([N_rays, N_samples + N_obj*N_samples_obj, 4])
            raw_sh = raw.shape
            # Predict RGB and density from background
            raw_bckg = network_query_fn(pts, viewdirs, network_fn)
            raw += tf.scatter_nd(id_z_vals_bckg, raw_bckg, raw_sh)
        else:
            raw = tf.zeros([N_rays, N_obj*N_samples_obj, 4])
            raw_sh = raw.shape

        if z_vals_in_o is not None and len(z_vals_in_o) != 0:
            # Loop for one model per object and no latent representations
            if latent_vector_dict is None:
                obj_id = tf.reshape(object_idx, obj_pose[..., 4].shape)
                for k, track_id in enumerate(object_y):
                    if track_id >= 0:
                        input_indices = tf.where(tf.equal(obj_id, k))
                        input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
                        model_name = 'model_obj_' + str(np.array(track_id).astype(np.int32))
                        # print('Hit', model_name, n_intersect, 'times.')
                        if model_name in object_network_fn_dict:
                            obj_network_fn = object_network_fn_dict[model_name]

                            inputs_obj_k = tf.gather_nd(inputs, input_indices)
                            viewdirs_obj_k = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0]) if N_samples_obj == 1 else \
                                tf.gather_nd(viewdirs_obj, input_indices[..., None,0, 0])

                            # Predict RGB and density from object model
                            raw_k = network_query_fn(inputs_obj_k, viewdirs_obj_k, obj_network_fn)

                            if n_intersect is not None:
                                # Arrange RGB and denisty from object models along the respective rays
                                raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj, 4]) # Project the network outputs to the corresponding ray
                                raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj, 4]) # Project to rays and object intersection order
                                raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh) # Reorder along z and ray
                            else:
                                raw_k = tf.scatter_nd(input_indices[:, 0][..., tf.newaxis], raw_k, [N_rays, N_samples, 4])

                            # Add RGB and density from object model to the background and other object predictions
                            raw += raw_k
            # Loop over classes c and evaluate each models f_c for all latent object describtor
            else:
                for c, class_type in enumerate(unique_classes.y):
                    # Ignore background class
                    if class_type >= 0:
                        input_indices = tf.where(tf.equal(class_id, c))
                        input_indices = tf.reshape(input_indices, [-1, N_samples_obj, 2])
                        model_name = 'model_class_' + str(int(np.array(class_type))).zfill(5)

                        if model_name in object_network_fn_dict:
                            obj_network_fn = object_network_fn_dict[model_name]

                            inputs_obj_c = tf.gather_nd(inputs, input_indices)

                            # Legacy version 2
                            # latent_vector = tf.concat([
                            #         latent_vector_dict['latent_vector_' + str(int(obj_id)).zfill(5)][tf.newaxis, :]
                            #         for obj_id in np.array(tf.gather_nd(obj_pose[..., 4], input_indices)).astype(np.int32).flatten()],
                            #         axis=0)
                            # latent_vector = tf.reshape(latent_vector, [inputs_obj_k.shape[0], inputs_obj_k.shape[1], -1])
                            # inputs_obj_k = tf.concat([inputs_obj_k, latent_vector], axis=-1)

                            # viewdirs_obj_k = tf.gather_nd(viewdirs_obj,
                            #                               input_indices[..., 0]) if N_samples_obj == 1 else \
                            #     tf.gather_nd(viewdirs_obj, input_indices)

                            viewdirs_obj_c = tf.gather_nd(viewdirs_obj, input_indices[..., None, 0])[:,0,:]

                            # Predict RGB and density from object model
                            raw_k = network_query_fn(inputs_obj_c, viewdirs_obj_c, obj_network_fn)

                            if n_intersect is not None:
                                # Arrange RGB and denisty from object models along the respective rays
                                raw_k = tf.scatter_nd(input_indices[:, :], raw_k, [n_intersect, N_samples_obj,
                                                                                   4])  # Project the network outputs to the corresponding ray
                                raw_k = tf.scatter_nd(intersection_map[:, :2], raw_k, [N_rays, N_obj, N_samples_obj,
                                                                                       4])  # Project to rays and object intersection order
                                raw_k = tf.scatter_nd(id_z_vals_obj, raw_k, raw_sh)  # Reorder along z in  positive ray direction
                            else:
                                raw_k = tf.scatter_nd(input_indices[:, 0][..., tf.newaxis], raw_k,
                                                      [N_rays, N_samples, 4])

                            # Add RGB and density from object model to the background and other object predictions
                            raw += raw_k
                        else:
                            print('No model ', model_name,' found')



    # raw_2 = render_mot_scene(pts, viewdirs, network_fn, network_query_fn,
    #                  inputs, viewdirs_obj, z_vals_in_o, n_intersect, object_idx, object_y, obj_pose,
    #                  unique_classes, class_id, latent_vector_dict, object_network_fn_dict,
    #                  N_rays,N_samples, N_obj, N_samples_obj,
    #                  obj_only=obj_only)

    # TODO: Reduce computation by removing 0 entrys
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        if sampling_method == 'planes' or sampling_method == 'planes_plus':
            pts, z_vals = plane_pts([rays_o, rays_d], [plane_bds, plane_normal, delta], id_planes, near,
                                    method=sampling_method)
        else:
            # Obtain additional integration times to evaluate based on the weights
            # assigned to colors in the coarse model.
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
            z_samples = tf.stop_gradient(z_samples)

            # Obtain all points to evaluate color, density at.
            z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        if use_time:
            pts = tf.concat([pts, time_stamp_fine], axis=-1)

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        if not sampling_method == 'planes':
            ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]
    # if latent_vector_dict is not None:
    #     ret['latent_loss'] = tf.reshape(latent_vector, [N_rays, N_samples_obj, -1])

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H,
           W,
           focal,
           chunk=1024*32,
           rays=None,
           c2w=None,
           obj=None,
           time_stamp=None,
           near=0.,
           far=1.,
           use_viewdirs=False,
           c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      obj: array of shape [batch_size, max_obj, n_obj_nodes]. Scene object's pose and propeties for each
      example in the batch
      time_stamp: bool. If True the frame will be taken into account as an additional input to the network
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        # rays = tf.random.shuffle(tf.concat([get_rays(H, W, focal, c2w)[0], get_rays(H, W, focal, c2w)[1]], axis=-1))
        # rays_o = rays[..., :3]
        # rays_d = rays[..., 3:]
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        if obj is not None:
            obj = tf.repeat(obj[None, ...], H*W, axis=0)
        if time_stamp is not None:
            time_stamp = tf.repeat(time_stamp[None, ...], H*W, axis=0)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    if time_stamp is not None:
        time_stamp = tf.cast(tf.reshape(time_stamp, [len(rays), -1]), dtype=tf.float32)
        rays = tf.concat([rays, time_stamp], axis=-1)

    if obj is not None:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction, scene objects)
        # obj = tf.cast(tf.reshape(obj, [obj.shape[0], obj.shape[1]*obj.shape[2]]), dtype=tf.float32)
        obj = tf.reshape(obj, [obj.shape[0], obj.shape[1] * obj.shape[2]])
        rays = tf.concat([rays, obj], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)
        # all_ret[k] = tf.reshape(all_ret[k], [k_sh[0], k_sh[1], -1])

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, obj=None, obj_meta=None, gt_imgs=None, savedir=None,
                render_factor=0, render_manipulation=None, rm_obj=None, time_stamp=None):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)

        if time_stamp is not None:
            time_st = time_stamp[i]
        else:
            time_st = None

        if obj is None:
            rgb, disp, acc, _ = render(
                H, W, focal, chunk=chunk, c2w=c2w[:3, :4], obj=None, time_stamp=time_st, **render_kwargs)

            rgbs.append(rgb.numpy())
            disps.append(disp.numpy())

            if i == 0:
                print(rgb.shape, disp.shape)

            if gt_imgs is not None and render_factor == 0:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                print(p)

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

            print(i, time.time() - t)

        else:
            # Manipulate scene graph edges
            # rm_obj = [3, 4, 8, 5, 12]
            render_set = manipulate_obj_pose(render_manipulation, np.array(obj), obj_meta, i, rm_obj=rm_obj)


            # Load manual generated scene graphs
            if render_manipulation is not None and 'handcraft' in render_manipulation:
                if str(i).zfill(3) + '.txt' in os.listdir(savedir):
                    print('Reloading', str(i).zfill(3) + '.txt')
                    render_set.pop()
                    loaded_obj_i = []
                    loaded_objs = np.loadtxt(os.path.join(savedir, str(i).zfill(3) + '.txt'))[:, :6]
                    loaded_objs[:, 5] = 0
                    loaded_objs[:, 4] = np.array([np.where(np.equal(obj_meta[:, 0], loaded_objs[j, 4])) for j in range(len(loaded_objs))])[:, 0, 0]
                    loaded_objs = tf.cast(loaded_objs, tf.float32)
                    loaded_obj_i.append(loaded_objs)
                    render_set.append(loaded_obj_i)
                if '02' in render_manipulation:
                    c2w = render_poses[36]
                if '03' in render_manipulation:
                    c2w = render_poses[20]
                if '04' in render_manipulation or '05' in render_manipulation:
                    c2w = render_poses[20]

            render_kwargs['N_obj'] = len(render_set[0][0])

            steps = len(render_set)
            for r, render_set_i in enumerate(render_set):
                t = time.time()
                j = steps * i + r
                obj_i = render_set_i[0]

                if obj_meta is not None:
                    obj_i_metadata = tf.gather(obj_meta, tf.cast(obj_i[:, 4], tf.int32),
                                               axis=0)
                    batch_track_id = obj_i_metadata[..., 0]

                    print("Next Frame includes Objects: ")
                    if batch_track_id.shape[0] > 1:
                        for object_tracking_id in np.array(tf.squeeze(batch_track_id)).astype(np.int32):
                            if object_tracking_id >= 0:
                                print(object_tracking_id)

                    obj_i_dim = obj_i_metadata[:, 1:4]
                    obj_i_label = obj_i_metadata[:, 4][:, tf.newaxis]
                    # xyz + roty
                    obj_i = obj_i[..., :4]

                    obj_i = tf.concat([obj_i, batch_track_id[..., None], obj_i_dim, obj_i_label], axis=-1)

                # obj_i = np.array(obj_i)
                # rm_ls_0 = [0, 1, 2,]
                # rm_ls_1 = [0, 1, 2]
                # rm_ls_2 = [0, 1, 2, 3, 5]
                # rm_ls = [rm_ls_0, rm_ls_1, rm_ls_2]
                # for k in rm_ls[i]:
                #     obj_i[k] = np.ones([9]) * -1

                rgb, disp, acc, _ = render(
                    H, W, focal, chunk=chunk, c2w=c2w[:3, :4], obj=obj_i, **render_kwargs)
                rgbs.append(rgb.numpy())
                disps.append(disp.numpy())

                if j == 0:
                    print(rgb.shape, disp.shape)

                if gt_imgs is not None and render_factor == 0:
                    p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                    print(p)

                if savedir is not None:
                    rgb8 = to8b(rgbs[-1])
                    filename = os.path.join(savedir, '{:03d}.png'.format(j))
                    imageio.imwrite(filename, rgb8)
                    if render_manipulation is not None:
                        if 'handcraft' in render_manipulation:
                            filename = os.path.join(savedir, '{:03d}.txt'.format(j))
                            np.savetxt(filename, np.array(obj_i), fmt='%.18e %.18e %.18e %.18e %.1e %.18e %.18e %.18e %.1e')


                print(j, time.time() - t)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    if args.obj_detection:
        trainable = False
    else:
        trainable = True

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    if args.use_time:
        input_ch += 1

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_color_head=input_ch_views, use_viewdirs=args.use_viewdirs, trainable=trainable)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_color_head=input_ch_views, use_viewdirs=args.use_viewdirs, trainable=trainable)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    models_dynamic_dict = None
    embedobj_fn = None
    latent_vector_dict = None if args.latent_size < 1 else {}
    latent_encodings = None if args.latent_size < 1 else {}
    if args.use_object_properties and not args.bckg_only:
        models_dynamic_dict = {}
        embedobj_fn, input_ch_obj = get_embedder(
            args.multires_obj, -1 if args.multires_obj == -1 else args.i_embed, input_dims=3)

        # Version a: One Network per object
        if args.latent_size < 1:
            input_ch = input_ch
            input_ch_color_head = input_ch_views
            # Don't add object location input for setting 1
            if args.object_setting != 1:
                input_ch_color_head += input_ch_obj
            # TODO: Change to number of objects in Frames
            for object_i in args.scene_objects:

                model_name = 'model_obj_' + str(int(object_i)) # .zfill(5)

                model_obj = init_nerf_model(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_color_head=input_ch_color_head, use_viewdirs=args.use_viewdirs,trainable=trainable)
                    # latent_size=args.latent_size)

                grad_vars += model_obj.trainable_variables
                models[model_name] = model_obj
                models_dynamic_dict[model_name] = model_obj

        # Version b: One Network for all similar objects of the same class
        else:
            input_ch = input_ch + args.latent_size
            input_ch_color_head = input_ch_views
            # Don't add object location input for setting 1
            if args.object_setting != 1:
                input_ch_color_head += input_ch_obj

            for obj_class in args.scene_classes:
                model_name = 'model_class_' + str(int(obj_class)).zfill(5)

                model_obj = init_nerf_model(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_color_head=input_ch_color_head,
                    # input_ch_shadow_head=input_ch_obj,
                    use_viewdirs=args.use_viewdirs, trainable=trainable)
                    # use_shadows=args.use_shadows,
                    # latent_size=args.latent_size)

                grad_vars += model_obj.trainable_variables
                models[model_name] = model_obj
                models_dynamic_dict[model_name] = model_obj

            for object_i in args.scene_objects:
                name = 'latent_vector_obj_'+str(int(object_i)).zfill(5)
                latent_vector_obj = init_latent_vector(args.latent_size, name)
                grad_vars.append(latent_vector_obj)

                latent_encodings[name] = latent_vector_obj
                latent_vector_dict[name] = latent_vector_obj

    # TODO: Remove object embedding function
    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        embedobj_fn=embedobj_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'N_samples_obj': args.N_samples_obj,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'object_network_fn_dict': models_dynamic_dict,
        'latent_vector_dict': latent_vector_dict if latent_vector_dict is not None else None,
        'N_obj': args.max_input_objects if args.use_object_properties and not args.bckg_only else False,
        'obj_only': args.obj_only,
        'obj_transparency': not args.obj_opaque,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'sampling_method': args.sampling_method,
        'use_time': args.use_time,
        'obj_location': False if args.object_setting == 1 else True,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    # render_kwargs_test['obj_only'] = False

    start = 0
    basedir = args.basedir
    expname = args.expname
    weights_path = None

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    elif args.model_library is not None and args.model_library != 'None':
        obj_ckpts = {}
        ckpts = []
        for f in sorted(os.listdir(args.model_library)):
            if 'model_' in f and 'fine' not in f and 'optimizer' not in f and 'obj' not in f:
                ckpts.append(os.path.join(args.model_library, f))
            if 'obj' in f and float(f[10:][:-11]) in args.scene_objects:
                obj_ckpts[f[:-11]] = (os.path.join(args.model_library, f))
    elif args.obj_only:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('_obj_' in f)]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f and 'obj' not in f and 'class' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload and (not args.obj_only or args.model_library):
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

        if models_dynamic_dict is not None:
            for model_dyn_name, model_dyn in models_dynamic_dict.items():
                if args.model_library:
                    ft_weights_obj = obj_ckpts[model_dyn_name]
                else:
                    ft_weights_obj = '{}'.format(ft_weights[:-16]) + \
                                     model_dyn_name + '_{}'.format(ft_weights[-10:])
                print('Reloading model from', ft_weights_obj, 'for', model_dyn_name[6:])
                model_dyn.set_weights(np.load(ft_weights_obj, allow_pickle=True))

        if latent_vector_dict is not None:
            for latent_vector_name, latent_vector in latent_vector_dict.items():
                ft_weights_obj = '{}'.format(ft_weights[:-16]) + \
                                     latent_vector_name + '_{}'.format(ft_weights[-10:])
                print('Reloading objects latent vector from', ft_weights_obj)
                latent_vector.assign(np.load(ft_weights_obj, allow_pickle=True))

    elif len(ckpts) > 0 and args.obj_only:
        ft_weights = ckpts[-1]
        start = int(ft_weights[-10:-4]) + 1
        ft_weights_obj_dir = os.path.split(ft_weights)[0]
        for model_dyn_name, model_dyn in models_dynamic_dict.items():
            ft_weights_obj = os.path.join(ft_weights_obj_dir, model_dyn_name + '_{}'.format(ft_weights[-10:]))
            print('Reloading model from', ft_weights_obj, 'for', model_dyn_name[6:])
            model_dyn.set_weights(np.load(ft_weights_obj, allow_pickle=True))

        if latent_vector_dict is not None:
            for latent_vector_name, latent_vector in latent_vector_dict.items():
                ft_weights_obj = os.path.join(ft_weights_obj_dir, latent_vector_name + '_{}'.format(ft_weights[-10:]))
                print('Reloading objects latent vector from', ft_weights_obj)
                latent_vector.assign(np.load(ft_weights_obj, allow_pickle=True))

        weights_path = ft_weights

    if args.model_library:
        start = 0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, latent_encodings, weights_path


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # Disabled and not implemented for Neural Scene Graphs
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--model_library", type=str, default=None,
                        help='specific weights npy file to load pretrained background and foreground models')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--sampling_method", type=str, default=None,
                        help='method to sample points along the ray options: None / lindisp / squaredist / plane')
    parser.add_argument("--crop_size", type=int, default=16,
                        help='size of crop image for second stage deblurring')
    parser.add_argument("--bckg_only", action='store_true',
                        help='removes rays associated with objects from the training set to train just the background model.')
    parser.add_argument("--obj_only", action='store_true',
                        help='Train object models on rays close to the objects only.')
    parser.add_argument("--use_inst_segm", action='store_true',
                        help='Use an instance segmentation map to select a subset from all sampled rays')
    parser.add_argument("--latent_size", type=int, default=0,
                        help='Size of the latent vector representing each of object of a class. If 0 no latent vector '
                             'is applied and a single representation per object is used.')
    parser.add_argument("--latent_balance", type=float, default=0.01,
                        help="Balance between image loss and latent loss")

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_samples_obj", type=int, default=3,
                        help='number of samples per ray and object')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_shadows", action='store_true',
                        help='use pose of an object to predict shadow opacity')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_obj", type=int, default=4,
                        help='log2 of max freq for positional encoding (3D object location + heading)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_time", action='store_true',
                        help='time parameter for nerf baseline version')
    parser.add_argument("--remove_frame", type=int, default=-1,
                        help="Remove the ith frame from the training set")
    parser.add_argument("--remove_obj", type=int, default=None,
                        help="Option to remove all pixels of an object from the training")

    # render flags
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument("--manipulate", type=str, default=None,
                        help='Renderonly manipulation argument')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / vkitti')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--training_factor", type=int, default=0,
                        help='downsample factor for all images during training')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # vkitti/kitti flags
    parser.add_argument("--first_frame", type=str, default=0,
                        help='specifies the beginning of a sequence if not the complete scene is taken as Input')
    parser.add_argument("--last_frame", type=str, default=None,
                        help='specifies the end of a sequence')
    parser.add_argument("--use_object_properties", action='store_true',
                        help='use pose and properties of visible objects as an Input')
    parser.add_argument("--object_setting", type=int, default=0,
                        help='specify which properties are used')
    parser.add_argument("--max_input_objects", type=int, default=20,
                        help='Max number of object poses considered by the network, will be set automatically')
    parser.add_argument("--scene_objects", type=list,
                        help='List of all objects in the trained sequence')
    parser.add_argument("--scene_classes", type=list,
                        help='List of all unique classes in the trained sequence')
    parser.add_argument("--obj_opaque", action='store_true',
                        help='Ray does stop after intersecting with the first object bbox if true')
    parser.add_argument("--single_obj", type=float, default=None,
                        help='Specify for sequential training.')
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Maximum scale for boxes to include shadows")
    parser.add_argument("--plane_type", type=str, default='uniform',
                        help='specifies how the planes are sampled')
    parser.add_argument("--near_plane", type=float, default=0.5,
                        help='specifies the distance from the last pose to the far plane')
    parser.add_argument("--far_plane", type=float, default=150.,
                        help='specifies the distance from the last pose to the far plane')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')

    # Object Detection through rendering
    parser.add_argument("--obj_detection", action='store_true',
                        help='Debug local')
    parser.add_argument("--frame_number", type=int, default=0,
                        help='Frame of the datadir which should be detected')

    # Local Debugging
    parser.add_argument("--debug_local", action='store_true',
                        help='Debug local')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    if args.obj_only and args.bckg_only:
        print('Object and background can not set as train only at the same time.')
        return

    if args.bckg_only or args.obj_only:
        # print('Deactivating object models to increase performance for training the background model only.')
        args.use_object_properties = True

    # Support first and last frame int
    starts = args.first_frame.split(',')
    ends = args.last_frame.split(',')
    if len(starts) != len(ends):
        print('Number of sequences is not defined. Using the first sequence')
        args.first_frame = int(starts[0])
        args.last_frame = int(ends[0])
    else:
        args.first_frame = [int(val) for val in starts]
        args.last_frame = [int(val) for val in ends]

    if args.dataset_type == 'kitti':
        # tracking2txt('../../CenterTrack/results/default_0006_results.json')

        images, poses, render_poses, hwf, i_split, visible_objects, objects_meta, render_objects, bboxes, \
        kitti_obj_metadata, time_stamp, render_time_stamp = \
            load_kitti_data(args.datadir,
                            selected_frames=[args.first_frame, args.last_frame] if args.last_frame else None,
                            use_obj=True,
                            row_id=True,
                            remove=args.remove_frame,
                            use_time=args.use_time,
                            exp=True if 'exp' in args.expname else False)
        print('Loaded kitti', images.shape,
              #render_poses.shape,
              hwf,
              args.datadir)

        if visible_objects is not None:
            args.max_input_objects = visible_objects.shape[1]
        else:
            args.max_input_objects = 0

        if args.render_only:
            visible_objects = render_objects

        i_train, i_val, i_test = i_split

        near = args.near_plane
        far = args.far_plane

        # Fix all persons at one position
        fix_ped_pose = False
        if fix_ped_pose:
            print('Pedestrians are fixed!')
            ped_poses = np.pad(visible_objects[np.where(visible_objects[..., 3] == 4)][:, 7:11], [[0, 0], [7, 3]])
            visible_objects[np.where(visible_objects[..., 3] == 4)] -= ped_poses
            visible_objects[np.where(visible_objects[..., 3] == 4)] += ped_poses[20]

        # Get SRN Poses, Images and Poses
        # work_dir = os.path.dirname(os.path.abspath(args.basedir))
        # srn_data_dir = os.path.join(work_dir, 'srn_data_' + args.expname)
        # srn_data_dir_train = srn_data_dir+'_train'
        # srn_data_dir_val = srn_data_dir + '_train_val'
        # if not os.path.exists(srn_data_dir_train):
        #     os.makedirs(srn_data_dir_train)
        #     os.makedirs(srn_data_dir_val)
        #     for dir in ['rgb', 'pose', 'intrinsics']:
        #         os.makedirs(os.path.join(srn_data_dir_train, dir))
        #         os.makedirs(os.path.join(srn_data_dir_val, dir))
        #
        # f = hwf[2]
        # c_x = hwf[1]/2.
        # c_y = hwf[0]/2.
        # id_val = 0
        #
        # for i, img in enumerate(images):
        #     frame_id = str(i).zfill(5)
        #     im = Image.fromarray((img*255).astype(np.uint8))
        #     im.save(os.path.join(srn_data_dir_train, 'rgb/'+frame_id+'.png'))
        #     np.savetxt(os.path.join(srn_data_dir_train, 'pose/' + frame_id + '.txt'),
        #                np.reshape(poses[i], [-1])[None], fmt='%.16f')
        #     np.savetxt(os.path.join(srn_data_dir_train, 'intrinsics/' + frame_id + '.txt'),
        #                np.array([f, 0.0, c_x, 0.0, f, c_y, 0.0, 0.0, 1.0], np.float32)[None], fmt='%.1f')
        #     if not i % 10:
        #         frame_id_val = str(id_val).zfill(5)
        #         im.save(os.path.join(srn_data_dir_val, 'rgb/' + frame_id_val + '.png'))
        #         np.savetxt(os.path.join(srn_data_dir_val, 'pose/' + frame_id_val + '.txt'),
        #                    np.reshape(poses[i], [-1])[None], fmt='%.16f')
        #         np.savetxt(os.path.join(srn_data_dir_val, 'intrinsics/' + frame_id_val + '.txt'),
        #                    np.array([f, 0.0, c_x, 0.0, f, c_y, 0.0, 0.0, 1.0], np.float32)[None], fmt='%.1f')
        #         id_val += 1
        #
        # print('Stored Image set for SRNs')


        # Get COLMAP formated poses
        # colmap_poses = poses[:, :3, :]
        # colmap_poses = np.concatenate([colmap_poses, np.repeat(np.array(hwf)[None], len(poses), axis=0)[..., None]], axis=2)
        # colmap_poses = np.reshape(colmap_poses, [-1,15])
        # colmap_poses = np.concatenate([colmap_poses, np.repeat(np.array([near, far])[None], len(poses), axis=0)], axis=1)
        # np.save(os.path.join(args.basedir, args.expname) +'/poses_bounds.npy', colmap_poses)

    elif args.dataset_type == 'vkitti':
        # TODO: Class by integer instead of hot-one-encoding for latent encoding in visible object
        images, instance_segm, poses, frame_id, render_poses, hwf, i_split, visible_objects, objects_meta, render_objects, bboxes = \
            load_vkitti_data(args.datadir,
                             selected_frames=[args.first_frame[0], args.last_frame[0]] if args.last_frame[0] >= 0 else -1,
                             use_obj=args.use_object_properties,
                             row_id=True if args.object_setting == 0 or args.object_setting == 1 else False,)
        render_time_stamp = None

        print('Loaded vkitti', images.shape,
              #render_poses.shape,
              hwf,
              args.datadir)
        if visible_objects is not None:
            args.max_input_objects = visible_objects.shape[1]
        else:
            args.max_input_objects = 0

        i_train, i_val, i_test = i_split

        near = args.near_plane
        far = args.far_plane

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Ploting Options for Debugging the Scene Graph
    plot_poses = False
    if args.debug_local and plot_poses:
        plot_kitti_poses(args, poses, visible_objects)

    # Cast intrinsics to right types
    np.linalg.norm(poses[:1, [0, 2], 3] - poses[1:, [0, 2], 3])
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Extract objects positions and labels
    if args.use_object_properties or args.bckg_only:
        obj_nodes, add_input_rows, obj_meta_ls, scene_objects, scene_classes = \
            extract_object_information(args, visible_objects, objects_meta)

        # obj_track_id_list = False if args.single_obj == None else [args.single_obj] #[4., 9.,, 3.] # [9.]

        if args.single_obj is not None:
            # Train only a single object
            args.scene_objects = [args.single_obj]
        else:
            args.scene_objects = scene_objects

        args.scene_classes = scene_classes

        n_input_frames = obj_nodes.shape[0]

        # Prepare object nodes [n_images, n_objects, H, W, add_input_rows, 3]
        obj_nodes = np.reshape(obj_nodes, [n_input_frames, args.max_input_objects * add_input_rows, 3])

        obj_meta_tensor = tf.cast(np.array(obj_meta_ls), tf.float32)

        if args.render_test:
            render_objects = obj_nodes[i_test]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf representation models
    render_kwargs_train, render_kwargs_test, start, grad_vars, models, latent_encodings, weights_path = create_nerf(
        args)

    if args.obj_only:
        print('removed bckg model for obj training')
        del grad_vars[:len(models['model'].trainable_variables)]
        models.pop('model')

    if args.ft_path is not None and args.ft_path != 'None':
        start = 0

    # Set bounds for point sampling along a ray
    if not args.sampling_method == 'planes' and not args.sampling_method == 'planes_plus':
        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
        }
    else:
        # TODO: Generalize for non front-facing scenarios
        plane_bds, plane_normal, plane_delta, id_planes, near, far = plane_bounds(
            poses, args.plane_type, near, far, args.N_samples)

        # planes = [plane_origin, plane_normal]
        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
            'plane_bds': tf.cast(plane_bds, tf.float32),
            'plane_normal': tf.cast(plane_normal, tf.float32),
            'id_planes': tf.cast(id_planes, tf.float32),
            'delta': tf.cast(plane_delta, tf.float32)
        }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if np.argwhere(n[:1,:,0]>0)only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        if args.manipulate is not None:
            testsavedir = testsavedir + '_' + args.manipulate

        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        # Select random from render_poses
        render_poses = render_poses[np.random.randint(0, len(render_poses) - 1, np.minimum(3, len(render_poses)))]

        rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              obj=obj_nodes if args.use_object_properties and not args.bckg_only else None,
                              obj_meta=obj_meta_tensor if args.use_object_properties else None,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor,
                              render_manipulation=args.manipulate, rm_obj=args.remove_obj,
                              time_stamp=render_time_stamp)
        print('Done rendering', testsavedir)
        if args.dataset_type == 'vkitti':
            rgbs = rgbs[:, 1:, ...]
            macro_block_size = 2
        else:
            macro_block_size = 16

        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=10, macro_block_size=macro_block_size)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)

    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)


    N_rand = args.N_rand
    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
    # for each pixel in the image. This stack() adds a new dimension.
    rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
    rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
    print('done, concats')
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)

    if not args.use_object_properties:
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        if args.use_time:
            time_stamp_train = np.stack([time_stamp[i]
                                   for i in i_train], axis=0)
            time_stamp_train = np.repeat(time_stamp_train[:, None, :], H*W, axis=0).astype(np.float32)
            rays_rgb = np.concatenate([rays_rgb, time_stamp_train], axis=1)

    else:
        print("adding object nodes to each ray")
        rays_rgb_env = rays_rgb
        input_size = 0

        obj_nodes = np.repeat(obj_nodes[:, :, np.newaxis, ...], W, axis=2)
        obj_nodes = np.repeat(obj_nodes[:, :, np.newaxis, ...], H, axis=2)

        obj_size = args.max_input_objects * add_input_rows
        input_size += obj_size
        # [N, ro+rd+rgb+obj_nodes, H, W, 3]
        rays_rgb_env = np.concatenate([rays_rgb_env, obj_nodes], 1)

        # [N, H, W, ro+rd+rgb+obj_nodes*max_obj, 3]
        # with obj_nodes [(x+y+z)*max_obj + (track_id+is_training+0)*max_obj]
        rays_rgb_env = np.transpose(rays_rgb_env, [0, 2, 3, 1, 4])
        rays_rgb_env = np.stack([rays_rgb_env[i]
                                 for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb+ obj_pose*max_obj, 3]
        rays_rgb_env = np.reshape(rays_rgb_env, [-1, 3+input_size, 3])

        rays_rgb = rays_rgb_env.astype(np.float32)
        del rays_rgb_env

        # get all rays intersecting objects
        if (args.bckg_only or args.obj_only or args.model_library is not None or args.use_object_properties): #and not args.debug_local:
            bboxes = None
            print(rays_rgb.shape)

            if args.use_inst_segm:
                # Ray selection from segmentation (early experiments)
                print('Using segmentation map')
                if not args.scene_objects:
                    rays_on_obj = np.where(instance_segm.flatten() > 0)[0]

                else:
                    # B) Single object per scene
                    rays_on_obj = []
                    for obj_track_id in args.scene_objects:
                        rays_on_obj.append(np.where(instance_segm.flatten() == obj_track_id+1)[0])
                    rays_on_obj = np.concatenate(rays_on_obj)
            elif bboxes is not None:
                # Ray selection from 2D bounding boxes (early experiments)
                print('Using 2D bounding boxes')
                rays_on_obj = get_bbox_pixel(bboxes, i_train, hwf)
            else:
                # Preferred option
                print('Using Ray Object Node intersections')
                rays_on_obj, rays_to_remove = get_all_ray_3dbox_intersection(rays_rgb, obj_meta_tensor,
                                                                             args.netchunk, local=args.debug_local,
                                                                             obj_to_remove=args.remove_obj)

            # Create Masks for background and objects to subsample the training batches
            obj_mask = np.zeros(len(rays_rgb), np.bool)
            obj_mask[rays_on_obj] = 1

            bckg_mask = np.ones(len(rays_rgb), np.bool)
            bckg_mask[rays_on_obj] = 0

            # Remove predefined objects from the scene
            if len(rays_to_remove) > 0 and args.remove_obj is not None:
                print('Removing obj ', args.remove_obj)
                # Remove rays from training set
                remove_mask = np.zeros(len(rays_rgb), np.bool)
                remove_mask[rays_to_remove] = 1
                obj_mask[remove_mask] = 0
                # Remove objects from graph
                rays_rgb = remove_obj_from_set(rays_rgb, np.array(obj_meta_ls), args.remove_obj)
                obj_nodes = np.reshape(np.transpose(obj_nodes, [0, 2, 3, 1, 4]), [-1, args.max_input_objects*2, 3])
                obj_nodes = remove_obj_from_set(obj_nodes, np.array(obj_meta_ls), args.remove_obj)
                obj_nodes = np.reshape(obj_nodes, [len(images), H, W, args.max_input_objects*2, 3])
                obj_nodes = np.transpose(obj_nodes, [0, 3, 1, 2, 4])

            # Debugging options to display selected rays/pixels
            debug_pixel_selection = False
            if args.debug_local and debug_pixel_selection:
                for i_smplimg in range(len(i_train)):
                    rays_rgb_debug = np.array(rays_rgb)
                    rays_rgb_debug[rays_on_obj, :] += np.random.rand(3) #0.
                    # rays_rgb_debug[remove_mask, :] += np.random.rand(3)
                    plt.figure()
                    img_sample = np.reshape(rays_rgb_debug[(H * W) * i_smplimg:(H * W) * (i_smplimg + 1), 2, :],
                                            [H, W, 3])
                    plt.imshow(img_sample)

                    # white_canvas = np.ones_like(rays_rgb_debug)
                    # white_canvas[rays_on_obj, :] = np.array([0., 0., 1.])
                    # white_sample = np.reshape(white_canvas[(H * W) * i_smplimg:(H * W) * (i_smplimg + 1), 2, :],
                    #                     [H, W, 3])
                    # white_sample = np.concatenate([white_sample, np.zeros([H, W, 1])], axis=2)
                    # white_sample = (white_sample*255).astype(np.uint8)
                    # white_sample[..., 3][np.where(white_sample[..., 1] < 1.)] = 255.
                    # plt.figure()
                    # plt.imshow(white_sample)
                    # Image.fromarray(white_sample).save('/home/julian/Desktop/sample.png')
                    # plt.arrow(0, H / 2, W, 0, color='red')
                    # plt.arrow(W / 2, 0, 0, H, color='red')
                    # plt.savefig('/home/julian/Desktop/debug_kitti_box/01/Figure_'+str(i_smplimg),)
                    # plt.close()

            if args.bckg_only:
                print('Removing objects from scene.')
                rays_rgb = rays_rgb[bckg_mask]
                print(rays_rgb.shape)
            elif args.obj_only and args.model_library is None or args.debug_local:
                print('Extracting objects from background.')
                rays_bckg = None
                rays_rgb = rays_rgb[obj_mask]
                print(rays_rgb.shape)
            else:
                rays_bckg = rays_rgb[bckg_mask]
                rays_rgb = rays_rgb[obj_mask]

            # Get Intersections per object and additional rays to have similar rays/object distributions VVVVV
            if not args.bckg_only:
                # # print(rays_rgb.shape)
                rays_rgb = resample_rays(rays_rgb, rays_bckg, obj_meta_tensor, objects_meta,
                                         args.scene_objects, scene_classes, args.chunk, local=args.debug_local)
            # Get Intersections per object and additional rays to have similar rays/object distributions AAAAA

    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')
    i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        batch_obj = None

        # Random over all images
        if not args.use_object_properties:
            # No object specific representations
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1+max_obj, 3*?]
        batch = tf.transpose(batch, [1, 0, 2])

        # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
        # target_s[n, rgb] = example_id, observed color.
        batch_rays, target_s, batch_dyn = batch[:2], batch[2], batch[3:]

        if args.use_time:
            batch_time = batch_dyn
        else:
            batch_time = None

        if args.use_object_properties:
            # batch_obj[N_rand, max_obj, properties+0]
            batch_obj_dyn = tf.reshape(tf.transpose(
                batch_dyn, [1, 0, 2]), [batch.get_shape()[1], args.max_input_objects, add_input_rows*3])


            # xyz + roty
            batch_obj = batch_obj_dyn[..., :4]

            # [N_rand, max_obj, trackID + label + model + color + Dimension]
            # Extract static nodes and edges (latent node, id, box size) for each object at each ray
            batch_obj_metadata = tf.gather(obj_meta_tensor, tf.cast(batch_obj_dyn[:, :, 4], tf.int32), axis=0)

            batch_track_id = batch_obj_metadata[:, :, 0]
            # TODO: For generalization later Give track ID in the beginning and change model name to track ID
            batch_obj = tf.concat([batch_obj, batch_track_id[..., None]], axis=-1)
            batch_dim = batch_obj_metadata[:, :, 1:4]
            batch_label = batch_obj_metadata[:, :, 4][..., tf.newaxis]

            batch_obj = tf.concat([batch_obj, batch_dim, batch_label], axis=-1)


            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0


        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays, obj=batch_obj, time_stamp=batch_time,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

            # Add loss for latent code
            if args.latent_size > 0:
                reg = 1/args.latent_balance    # 1/0.01
                latent_reg = latentReg(list(render_kwargs_train['latent_vector_dict'].values()), reg)
                loss += latent_reg

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0
        #####           end            #####

        # Rest is logging

        def save_weights(weights, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, weights)
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k].get_weights(), k, i)
            if args.latent_size > 0:
                for k in latent_encodings:
                    save_weights(latent_encodings[k].numpy(), k, i)

        if i % args.i_print == 0 or i < 10:
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                # if args.N_importance > 0:
                #     tf.contrib.summary.scalar('psnr0', psnr0)
                # else:
                #     tf.contrib.summary.histogram('tran', trans)

                if args.latent_size > 0:
                    for latent_vector_sum in list(render_kwargs_train['latent_vector_dict'].values()):
                        tf.contrib.summary.histogram(
                            latent_vector_sum.name,
                            latent_vector_sum.value(),
                        )

            if i % args.i_img == 0 and not i == 0: # and not args.debug_local:

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3, :4]
                time_st = time_stamp[img_i] if args.use_time else None

                if args.use_object_properties:
                    obj_i = obj_nodes[img_i, :, 0, 0, ...]
                    obj_i = tf.cast(obj_i, tf.float32)
                    obj_i = tf.reshape(obj_i, [args.max_input_objects, obj_i.shape[0] // args.max_input_objects * 3])

                    obj_i_metadata = tf.gather(obj_meta_tensor, tf.cast(obj_i[:, 4], tf.int32),
                                                       axis=0)
                    batch_track_id = obj_i_metadata[..., 0]
                    obj_i_dim = obj_i_metadata[:, 1:4]
                    obj_i_label = obj_i_metadata[:, 4][:, tf.newaxis]

                    # xyz + roty
                    obj_i = obj_i[..., :4]
                    obj_i = tf.concat([obj_i, batch_track_id[..., None], obj_i_dim, obj_i_label], axis=-1)

                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, obj=obj_i,
                                                    **render_kwargs_test)
                else:
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, time_stamp=time_st,
                                                    **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))
                
                # Save out the validation image for Tensorboard-free monitoring
                testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                if i==0 or not os.path.exists(testimgdir):
                    os.makedirs(testimgdir, exist_ok=True)
                imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()