import os
from PIL import Image
import tensorflow as tf
import numpy as np
import lpips_tf
from cv2 import *
from matplotlib import pyplot as plt

calcOpticalFlow = cv2.calcOpticalFlowFarneback
rgb2gray = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
tLPmodel = lambda img_0, img_1: lpips_tf.lpips(img_0, img_1, model='net-lin', net='alex')


def crop_8x8(img):
    ori_h = img.shape[0]
    ori_w = img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32

    while (h > ori_h - 16):
        h = h - 32
    while (w > ori_w - 16):
        w = w - 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y + h, x:x + w]
    return crop_img, y, x

# Temporal Optical Flow https://ge.in.tum.de/publications/2019-tecogan-chu/
def tOF(pre_out_img_gray, out_img_gray, pre_tar_img_gray, tar_img_gray):

    target_OF = cv2.calcOpticalFlowFarneback(pre_tar_img_gray, tar_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    output_OF = cv2.calcOpticalFlowFarneback(pre_out_img_gray, out_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    target_OF, ofy, ofx = crop_8x8(target_OF)
    output_OF, ofy, ofx = crop_8x8(output_OF)
    OF_diff = np.absolute(target_OF - output_OF)

    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm
    # OF_diff, ofy, ofx = crop_8x8(OF_diff)
    # list_dict["tOF"].append(OF_diff.mean())
    # msg += "tOF %02.2f, " % (list_dict["tOF"][-1])

    return OF_diff

# Tempotal LPIPS https://ge.in.tum.de/publications/2019-tecogan-chu/
def tLP100(pre_out_img, out_img, pre_tar_img, tar_img):
    dist0t = tLPmodel(pre_tar_img, tar_img)
    dist1t = tLPmodel(pre_out_img, out_img)
    # print ("tardis %f, outdis %f" %(dist0t, dist1t))
    dist01t = np.absolute(dist0t - dist1t) * 100.0  ##########!!!!!
    # list_dict["tLP100"].append(dist01t[0])
    # msg += ", tLPx100 %02.2f" % (dist01t[0])
    return dist01t


def load_image(fname):
    img = Image.open(fname)

    return tf.keras.preprocessing.image.img_to_array(img)/250.0


def load_metric_fn(method):
    method_fns = {}
    metric_dict = {}
    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)

    session = tf.Session()

    if method == 'all':
        method = ['psnr', 'ssim', 'lpips', 'tLP100', 'tOF']
    elif method == 'temporal':
        method = ['tLP100', 'tOF']
    else:
        method = [method]

    for fn_name in method:
        if fn_name == 'psnr':
            fn = session.make_callable(
                tf.image.psnr(image0_ph, image1_ph, max_val=1.),
                [image0_ph, image1_ph])
        if fn_name == 'ssim':
            fn = session.make_callable(
                tf.image.ssim(image0_ph, image1_ph, max_val=1.),
                [image0_ph, image1_ph])
        if fn_name == 'lpips':
            fn = session.make_callable(
                lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='vgg'),
                [image0_ph, image1_ph])
        if fn_name == 'tLP100':
            fn = session.make_callable(
                lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex'),
                [image0_ph, image1_ph]
            )
        if fn_name == 'tOF':
            fn = session.make_callable(
                lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex'),
                [image0_ph, image1_ph]
            )
        # Add new methods here!

        method_fns[fn_name.upper()] = fn
        metric_dict[fn_name.upper()] = []

    return method_fns, metric_dict


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='temporal',
                        help='method name: all, lpips, psnr, ssim, temporal')
    parser.add_argument("--gt_dir", type=str, default='/home/julian/Desktop/renderings_kitti/0006/gt_0006_65_120',
                        help='GT images directory')
    parser.add_argument("--render_dir", type=str, default='/home/julian/Desktop/renderings_kitti/0006/latent_jointly_04_really_big_dense_boxes/renderonly_path_990001',
                        help='Rendering dir')
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of image pairs evaluated at the same time")

    return parser


def main():
    """
    Calculate lpips, psnr, ssim, tLP and tOF
    """
    parser = config_parser()
    args = parser.parse_args()

    center_cut = False
    if 'srn' in args.render_dir:
        center_cut = True

    gt_img_names = [img_name for img_name in sorted(os.listdir(args.gt_dir)) if img_name[-3:] == 'png']
    renderings_names = [img_name for img_name in sorted(os.listdir(args.render_dir)) if img_name[-3:] == 'png']

    method_fns, metric_dict = load_metric_fn(args.method)

    if not len(gt_img_names) == len(renderings_names):
        print('Renderings and GT images do not match!!')
    elif args.method == 'temporal':
        pass
    else:
        n = len(gt_img_names)
        gt_imgs = np.concatenate([
            load_image(os.path.join(args.gt_dir, gt_img_names[i]))[None] for i in range(n)])
        if center_cut:
            gt_imgs = gt_imgs[:, :, 1242//2-375//2:1242//2+375//2+1, :]
        renderings = np.concatenate([
            load_image(os.path.join(args.render_dir, renderings_names[i]))[None] for i in range(n)])

        # image 2 image comparison
        for name, method in method_fns.items():
            k = 0
            while k <= n:
                value = method(gt_imgs[k:(k+1)*args.batch_size], renderings[k:(k+1)*args.batch_size])
                metric_dict[name].append(value)
                k += args.batch_size

    # temporal metrics for two adjacent frames
    if args.method == 'temporal' or args.method == 'all':
        n = len(gt_img_names)

        gt_imgs = np.concatenate([
            load_image(os.path.join(args.gt_dir, gt_img_names[i]))[None] for i in range(n)])
        if center_cut:
            gt_imgs = gt_imgs[:, :, 1242//2-375//2:1242//2+375//2+1, :]
        renderings = np.concatenate([
            load_image(os.path.join(args.render_dir, renderings_names[i]))[None] for i in range(n)])

        tLP_model = method_fns['TLP100']

        k = 1
        while k < n:
            output_img = renderings[k, ...]
            target_img = gt_imgs[k, ...]

            # tOF
            output_grey = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
            target_grey = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

            if (k >= 2):  # temporal metrics
                target_OF = cv2.calcOpticalFlowFarneback(pre_tar_grey, target_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                output_OF = cv2.calcOpticalFlowFarneback(pre_out_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                target_OF, ofy, ofx = crop_8x8(target_OF)
                output_OF, ofy, ofx = crop_8x8(output_OF)
                OF_diff = np.absolute(target_OF - output_OF)

                OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm
                # OF_diff, ofy, ofx = crop_8x8(OF_diff)
                # list_dict["tOF"].append(OF_diff.mean())
                # msg += "tOF %02.2f, " % (list_dict["tOF"][-1])
                metric_dict['TOF'].append([OF_diff.mean()])
                # print(OF_diff.mean())

            pre_out_grey = output_grey
            pre_tar_grey = target_grey

            # tLP100
            target_img, ofy, ofx = crop_8x8(target_img)
            output_img, ofy, ofx = crop_8x8(output_img)

            # img0 = util.im2tensor(target_img)  # RGB image from [-1,1]
            img0 = target_img
            # img1 = util.im2tensor(output_img)
            img1 = output_img

            if "TLP100" in method_fns.keys() and (k >= 2):  # tLP, temporal metrics

                dist0t = tLP_model(pre_img0, img0)
                dist1t = tLP_model(pre_img1, img1)
                # print ("tardis %f, outdis %f" %(dist0t, dist1t))
                dist01t = np.absolute(dist0t - dist1t) * 100.0  ##########!!!!!
                # list_dict["tLP100"].append(dist01t[0])
                # msg += ", tLPx100 %02.2f" % (dist01t[0])
                metric_dict['TLP100'].append([dist01t])
                # print(dist01t)

            pre_img0 = img0
            pre_img1 = img1

            k += 1

        # msg += ", crop (%d, %d)" % (ofy, ofx)
        # print(msg)


    if metric_dict:
        for name, val_ls in metric_dict.items():
            # print(name)
            # print(val_ls)
            val_avg = np.mean(np.concatenate(val_ls))
            print(name, val_avg)


if __name__ == '__main__':
    main()