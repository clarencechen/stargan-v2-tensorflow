"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from StarGAN_v2 import StarGAN_v2
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StarGAN_v2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--merge', type=str2bool, default=True, help='In test phase, merge reference-guided image result or not')
    parser.add_argument('--merge_size', type=int, default=0, help='merge size matching number')
    parser.add_argument('--dataset', type=str, default='celeba_hq_gender', help='dataset_name')
    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--ds_iter', type=int, default=100000, help='Number of iterations to optimize diversity sensitive loss')

    parser.add_argument('--batch_size', type=int, default=8, help='The batch size for each replica')  # each gpu
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--num_style', type=int, default=5, help='Number of generated images per domain during sampling')

    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='The learning rate')
    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay value')

    parser.add_argument('--adv_weight', type=float, default=1, help='The weight of Adversarial loss')
    parser.add_argument('--sty_weight', type=float, default=1, help='Weight for style reconstruction loss')
    parser.add_argument('--ds_weight', type=float, default=1, help='Weight for diversity sensitive loss') # 2 for animal
    parser.add_argument('--cyc_weight', type=float, default=1, help='Weight for cyclic consistency loss')
    parser.add_argument('--r1_weight', type=float, default=1, help='Weight for R1 regularization')

    parser.add_argument('--gan_type', type=str, default='gan-gp', help='gan / lsgan / gan-gp / hinge')
    parser.add_argument('--sn', action='store_true', help='using spectral norm')

    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', action='store_true', help='Image augmentation use or not')

    parser.add_argument('--use_tfrecord', action='store_true', help='Flag to use tfrecord packed dataset')
    parser.add_argument('--num_shards', type=int, default=1, help='Number of shards in divided tfrecord dataset')
    parser.add_argument('--use_tpu', action='store_true', help='Flag to use available TPU device')

    parser.add_argument('--save_dir', type=str, default='.', help='Directory name to save checkpoints/logs/outputs')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --save_dir
    check_folder(args.save_dir)

    # --epoch
    try:
        assert args.iteration >= 1
    except:
        print('number of iterations must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --use_tpu
    if args.use_tpu:
        try:
            assert args.use_tfrecord
        except:
            print('must use tfrecord dataset when training on tpu device')

    return args

"""main"""
def main():

    args = parse_args()
    
    # Find TPU device if possible
    if args.use_tpu :
        try :
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            tpu = None
    
    else :
        tpu = None

    # TPUStrategy for distributed training
    if tpu :
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    
    else :
        automatic_gpu_usage()
        strategy = tf.distribute.get_strategy()

    gan = StarGAN_v2(args, strategy)

    # build graph
    gan.build_model()


    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    else :
        gan.test(args.merge, args.merge_size)
        print(" [*] Test finished!")



if __name__ == '__main__':
    main()
