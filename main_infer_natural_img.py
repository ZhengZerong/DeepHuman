from __future__ import print_function, absolute_import, division

import os
import zipfile
import time
import pynvml
import numpy as np
import tensorflow as tf
from TrainerNormal import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pynvml.nvmlInit()


def waitgpu(empty_thres_duration=10):
    empty_flag = 0
    while True:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage_percent = float(meminfo.used)/float(meminfo.total)
        if usage_percent < 0.1:
            if empty_flag >= empty_thres_duration:   # empty for 5 second
                break
            empty_flag += 1
            time.sleep(1)
            continue
        empty_flag = 0
        print('GPU is busy right now....waiting....')
        print('meminfo.used/meminfo.total = %f' % usage_percent)
        time.sleep(np.random.randint(5, 15))


def main(img_folder, img_prefix, zip_intermedia_results=True):
    waitgpu()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    trainer = Trainer(sess)
    trainer.test(img_folder, [img_prefix],
                 './results/results_final_19_09_30_10_29_33')
    os.system('python2 ./DataUtil/NormalFusion.py --volume_file %s --normal_file %s'
              % (os.path.join(img_folder, img_prefix + '_volume_out.mat'),
                 os.path.join(img_folder, img_prefix + '_normal_0.png')))

    suffixes = ['__normal_0.png', '__normal_1.png', '__normal_2.png', '__normal_3.png',
                '__normal_orig_0.png', '__normal_orig_1.png', '__normal_orig_2.png',
                '__normal_orig_3.png', '__volume_out.mat']
    if zip_intermedia_results:
        z = zipfile.ZipFile(img_dir[:-4] + '_intermediate_infer.zip', 'w')
        for suffix in suffixes:
            z.write(img_dir[:-4] + suffix)
            os.remove(img_dir[:-4] + suffix)
        z.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='path to image file')
    args = parser.parse_args()
    img_dir = args.file
    if not (img_dir.endswith('.png') or img_dir.endswith('.jpg')):
        print('Unsupport image format!!!')
        raise ValueError

    img_folder, img_name = os.path.split(img_dir)
    main(img_folder, img_name[:-4] + '_')
