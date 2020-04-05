from __future__ import print_function, absolute_import, division

import os
import time
import numpy as np
import tensorflow as tf
from TrainerNormal import Trainer
import CommonUtil as util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    util.safe_mkdir('./results')
    util.safe_mkdir('./debug')

    # please define your own split here
    img_total_num = 28000
    split = 0.8
    indices = np.asarray(range(img_total_num))
    testing_flag = (indices > split*max_idx)
    testing_inds = indices[testing_flag]
    training_inds = indices[np.logical_not(testing_flag)]

    testing_inds = testing_inds.tolist()
    training_inds = training_inds.tolist()
    np.random.shuffle(testing_inds)
    np.random.shuffle(training_inds)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    time_str = time.strftime('%y_%m_%d_%H_%M_%S')

    trainer = Trainer(sess)
    trainer.train('./TrainingDataPreparation/synthetic_dataset_final', training_inds, testing_inds,
                  results_dir='./results/results_final_' + time_str,  # directory to stored the results
                  graph_dir='./results/graph_final_' + time_str,  # directory as tensorboard working space
                  batch_size=4,  # batch size
                  epoch_num=12,  # epoch number
                  first_channel=8,
                  bottle_width=4,
                  dis_reps=1,
                  mode='retrain',
                  pre_model_dir=None)


if __name__ == '__main__':
    main()
