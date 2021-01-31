"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from utils import *
import time
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device

from glob import glob
from tqdm import tqdm
from networks import *
from copy import deepcopy
import PIL.Image

class StarGAN_v2():
    def __init__(self, args, strategy):
        super(StarGAN_v2, self).__init__()

        self.model_name = 'StarGAN_v2'
        self.strategy = strategy
        self.phase = args.phase

        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag
        self.use_tfrecord = args.use_tfrecord
        self.num_shards = args.num_shards

        self.ds_iter = args.ds_iter
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size * strategy.num_replicas_in_sync
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.lr = args.lr * strategy.num_replicas_in_sync
        self.f_lr = args.f_lr * strategy.num_replicas_in_sync
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.ema_decay = args.ema_decay

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight_init = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight

        """ Generator """
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim

        """ Mapping Network """
        self.hidden_dim = args.hidden_dim

        """ Discriminator """
        self.sn = args.sn

        self.sample_dir = os.path.join(args.save_dir, self.model_dir, 'samples')
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(args.save_dir, self.model_dir, 'checkpoint')
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.save_dir, self.model_dir, 'tensorboard')
        check_folder(self.log_dir)

        self.result_dir = os.path.join(args.save_dir, self.model_dir, 'results')
        check_folder(self.result_dir)

        self.use_tpu = args.use_tpu

        if self.use_tfrecord:
            dataset_path = 'gs://celeba-hq-dataset/' # modify later to accomodate multiple datasets
            self.domain_list = ['male', 'female']
        else:
            dataset_path = os.path.join('./data', self.dataset_name)
            self.domain_list = sorted([os.path.basename(x) for x in glob(self.dataset_path + '/*')])
        
        self.dataset_path = os.path.join(dataset_path, 'train')
        self.test_dataset_path = os.path.join(dataset_path, 'test')
        self.num_domains = len(self.domain_list)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# domain_list : ", self.domain_list)

        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# ds iteration : ", self.ds_iter)

        print()

        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)
        print("# style_dim : ", self.style_dim)

        print()

        print("##### Mapping Network #####")
        print("# hidden_dim : ", self.hidden_dim)

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""
            img_class = Image_data(self.img_size, self.img_ch, self.dataset_path, self.augment_flag, self.batch_size, self.latent_dim)

            dataset_num = 28000 # modify later to accomodate multiple datasets
            print("Dataset number : ", dataset_num)

            if self.use_tfrecord:
                filenames = ['{}_part_{}'.format(os.path.join(self.dataset_path, 'data.tfrecord.gzip'), i) for i in range(self.num_shards)]
                img_and_domain = tf.data.TFRecordDataset(filenames, compression_type='GZIP', num_parallel_reads=self.num_shards)
            else:   
                images, domains = build_filename_list(self.dataset_path, self.domain_list)
                img_and_domain = tf.data.Dataset.from_tensor_slices((images, domains))
                gpu_device = '/gpu:0'

            img_and_domain = img_and_domain.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()

            if self.use_tfrecord:
                img_and_domain = img_and_domain.map(img_class.parse_function, num_parallel_calls=AUTOTUNE)
            else:
                img_and_domain = img_and_domain.map(img_class.read_function, num_parallel_calls=AUTOTUNE)

            if self.augment_flag:
                img_and_domain = img_and_domain.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE)

            img_and_domain = img_and_domain.batch(self.batch_size, drop_remainder=True)
            img_and_domain = img_and_domain.map(img_class.inject_z, num_parallel_calls=AUTOTUNE)

            if self.use_tpu:
                img_and_domain = img_and_domain.prefetch(buffer_size=AUTOTUNE)
                img_and_domain = self.strategy.experimental_distribute_dataset(img_and_domain)

            else:
                img_and_domain = img_and_domain.apply(prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))

            self.img_and_domain_iter = iter(img_and_domain)

            with self.strategy.scope():
                """ Network """
                self.generator = Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim, sn=self.sn, name='Generator')
                self.mapping_network = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=self.sn, name='MappingNetwork')
                self.style_encoder = StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim, sn=self.sn, name='StyleEncoder')
                self.discriminator = Discriminator(self.img_size, self.num_domains, max_conv_dim=self.hidden_dim, sn=self.sn, name='Discriminator')

                self.generator_ema = deepcopy(self.generator)
                self.mapping_network_ema = deepcopy(self.mapping_network)
                self.style_encoder_ema = deepcopy(self.style_encoder)

                """ Finalize model (build) """
                x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
                y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
                z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
                s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

                _ = self.mapping_network([z, y])
                _ = self.mapping_network_ema([z, y])
                _ = self.style_encoder([x, y])
                _ = self.style_encoder_ema([x, y])
                _ = self.generator([x, s])
                _ = self.generator_ema([x, s])
                _ = self.discriminator([x, y])


                """ Optimizer """
                self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
                self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
                self.f_optimizer = tf.keras.optimizers.Adam(learning_rate=self.f_lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)
                self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-08)


                """ Checkpoint """
                self.ckpt = tf.train.Checkpoint(generator=self.generator, generator_ema=self.generator_ema,
                                                mapping_network=self.mapping_network, mapping_network_ema=self.mapping_network_ema,
                                                style_encoder=self.style_encoder, style_encoder_ema=self.style_encoder_ema,
                                                discriminator=self.discriminator,
                                                g_optimizer=self.g_optimizer, e_optimizer=self.e_optimizer, f_optimizer=self.f_optimizer,
                                                d_optimizer=self.d_optimizer)
                self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)
                self.start_iteration = 0

                if self.manager.latest_checkpoint:
                    self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                    self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                    print('Latest checkpoint restored!!')
                    print('start iteration : ', self.start_iteration)
                else:
                    print('Not restoring from saved checkpoint')

        else:
            """ Test """
            """ Network """
            self.generator_ema = Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim, sn=False, name='Generator')
            self.mapping_network_ema = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False, name='MappingNetwork')
            self.style_encoder_ema = StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim, sn=False, name='StyleEncoder')

            """ Finalize model (build) """
            x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
            y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
            z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
            s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

            _ = self.mapping_network_ema([z, y])
            _ = self.style_encoder_ema([x, y])
            _ = self.generator_ema([x, s])

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(generator_ema=self.generator_ema,
                                            mapping_network_ema=self.mapping_network_ema,
                                            style_encoder_ema=self.style_encoder_ema)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    @tf.function
    def g_train_step(self, x_real, y_org, y_trgs, ds_weight, z_trgs=None, x_refs=None):
        with tf.GradientTape(persistent=True) as g_tape:
            y_trg, y_trg2 = y_trgs
            if z_trgs is not None:
                z_trg, z_trg2 = z_trgs
            if x_refs is not None:
                x_ref, x_ref2 = x_refs

            # adversarial loss
            if z_trgs is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else:
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg])
            fake_logit = self.discriminator([x_fake, y_trg])
            g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit) / self.batch_size

            # style reconstruction loss
            s_pred = self.style_encoder([x_fake, y_trg])
            g_sty_loss = self.sty_weight * L1_loss(s_pred, s_trg, batch_size=self.batch_size)

            # diversity sensitive loss
            if z_trgs is not None:
                s_trg2 = self.mapping_network([z_trg2, y_trg2])
            else:
                s_trg2 = self.style_encoder([x_ref2, y_trg])

            x_fake2 = self.generator([x_real, s_trg2])
            x_fake2 = tf.stop_gradient(x_fake2)
            g_ds_loss = -ds_weight * L1_loss(x_fake, x_fake2, batch_size=self.batch_size)

            # cycle-consistency loss
            s_org = self.style_encoder([x_real, y_org])
            x_rec = self.generator([x_fake, s_org])
            g_cyc_loss = self.cyc_weight * L1_loss(x_rec, x_real, batch_size=self.batch_size)

            regular_loss = regularization_loss(self.generator)

            g_loss = g_adv_loss + g_sty_loss + g_ds_loss + g_cyc_loss + regular_loss

        g_train_variable = self.generator.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        if z_trgs is not None:
            f_train_variable = self.mapping_network.trainable_variables
            e_train_variable = self.style_encoder.trainable_variables

            f_gradient = g_tape.gradient(g_loss, f_train_variable)
            e_gradient = g_tape.gradient(g_loss, e_train_variable)

            self.f_optimizer.apply_gradients(zip(f_gradient, f_train_variable))
            self.e_optimizer.apply_gradients(zip(e_gradient, e_train_variable))

        return g_adv_loss, g_sty_loss, g_ds_loss, g_cyc_loss, g_loss

    @tf.function
    def d_train_step(self, x_real, y_org, y_trg, z_trg=None, x_ref=None):
        with tf.GradientTape() as d_tape:

            if z_trg is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else: # x_ref is not None
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg])

            real_logit = self.discriminator([x_real, y_org])
            fake_logit = self.discriminator([x_fake, y_trg])

            d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit) / self.batch_size

            if self.gan_type == 'gan-gp':
                d_adv_loss += self.r1_weight * r1_gp_req(self.discriminator, x_real, y_org) / self.batch_size

            regular_loss = regularization_loss(self.discriminator)

            d_loss = d_adv_loss + regular_loss

        d_train_variable = self.discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))

        return d_adv_loss, d_loss

    @tf.function
    def train_combined_step(self, x_real, y_org, x_refs, y_trgs, z_trgs, ds_weight):

        # update discriminator
        d_out_lat = self.strategy.run(self.d_train_step, args=(x_real, y_org, y_trgs[0]), kwargs={'z_trg': z_trgs[0]})
        d_out_ref = self.strategy.run(self.d_train_step, args=(x_real, y_org, y_trgs[0]), kwargs={'x_ref': x_refs[0]})

        # update generator
        g_out_lat = self.strategy.run(self.g_train_step, args=(x_real, y_org, y_trgs, ds_weight), kwargs={'z_trgs': z_trgs})
        g_out_ref = self.strategy.run(self.g_train_step, args=(x_real, y_org, y_trgs, ds_weight), kwargs={'x_refs': x_refs})

        # compute moving average of network parameters
        moving_average(self.generator, self.generator_ema, beta=self.ema_decay)
        moving_average(self.mapping_network, self.mapping_network_ema, beta=self.ema_decay)
        moving_average(self.style_encoder, self.style_encoder_ema, beta=self.ema_decay)

        # aggregate and return losses over all replicas
        return [[self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None) for loss in losses] for losses in (d_out_lat, d_out_ref, g_out_lat, g_out_ref)]

    def train(self):

        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.generator.summary()
        self.discriminator.summary()
        self.mapping_network.summary()
        self.style_encoder.summary()

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            # decay weight for diversity sensitive loss as a tf.Tensor
            ds_weight = tf.maximum(0.0, self.ds_weight_init -(self.ds_weight_init / self.ds_iter) * idx)
            
            x_real, y_org, _ = next(self.img_and_domain_iter)
            x_ref, y_trg, z_trg = next(self.img_and_domain_iter)
            x_ref2, y_trg2, z_trg2 = next(self.img_and_domain_iter)

            loss_package = self.train_combined_step(x_real, y_org, (x_ref, x_ref2), (y_trg, y_trg2), (z_trg, z_trg2), ds_weight)

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g/latent/adv_loss', loss_package[2][0], step=idx)
                tf.summary.scalar('g/latent/sty_loss', loss_package[2][1], step=idx)
                tf.summary.scalar('g/latent/ds_loss', loss_package[2][2], step=idx)
                tf.summary.scalar('g/latent/cyc_loss', loss_package[2][3], step=idx)
                tf.summary.scalar('g/latent/loss', loss_package[2][4], step=idx)

                tf.summary.scalar('g/ref/adv_loss', loss_package[3][0], step=idx)
                tf.summary.scalar('g/ref/sty_loss', loss_package[3][1], step=idx)
                tf.summary.scalar('g/ref/ds_loss', loss_package[3][2], step=idx)
                tf.summary.scalar('g/ref/cyc_loss', loss_package[3][3], step=idx)
                tf.summary.scalar('g/ref/loss', loss_package[3][4], step=idx)

                tf.summary.scalar('g/ds_weight', ds_weight, step=idx)

                tf.summary.scalar('d/latent/adv_loss', loss_package[0][0], step=idx)
                tf.summary.scalar('d/latent/loss', loss_package[0][1], step=idx)

                tf.summary.scalar('d/ref/adv_loss', loss_package[1][0], step=idx)
                tf.summary.scalar('d/ref/loss', loss_package[1][1], step=idx)

            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:


                latent_fake_save_path = '{}/latent_{:07d}.jpg'.format(self.sample_dir, idx + 1)
                ref_fake_save_path = '{}/ref_{:07d}.jpg'.format(self.sample_dir, idx + 1)

                latent_out = self.latent_canvas(x_real, z_trg)
                refer_out = self.refer_canvas(x_real, x_ref, y_trg)

                latent_fobj = tf.io.gfile.GFile(latent_fake_save_path, mode='wb')
                refer_fobj = tf.io.gfile.GFile(ref_fake_save_path, mode='wb')

                latent_out.save(latent_fobj, format='jpeg')
                refer_out.save(refer_fobj, format='jpeg')

            print("iter: {0:6d}/{1:6d} time: {2:4.4f} d_loss: {3:.8f}, g_loss: {4:.8f}, sty_loss: {5:.8f}, ds_loss: {6:.8f}".format(
                idx, self.iteration, time.time() - iter_start_time,
                loss_package[0][-1] + loss_package[1][-1],
                loss_package[2][-1] + loss_package[3][-1],
                loss_package[2][1] + loss_package[3][1],
                loss_package[2][2] + loss_package[3][2]
                ))
            del x_real, y_org, x_ref, y_trg, x_ref2, y_trg2, z_trg, z_trg2, loss_package

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

        print("Total train time: {4.4f}".format(time.time() - start_time))

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}{}".format(self.model_name, self.dataset_name, self.gan_type, sn)

    def refer_canvas(self, x_real, x_ref, y_trg):
        canvas = PIL.Image.new('RGB', (self.img_size * (self.batch_size + 1) + 10, self.img_size * (self.batch_size + 1) + 10),
                               'white')

        if self.strategy.num_replicas_in_sync > 1:
            x_real_post = postprocess_images(tf.concat(list(x_real.values), axis=0)).numpy()
            x_ref_post = postprocess_images(tf.concat(list(x_ref.values), axis=0)).numpy()
        else:
            x_real_post = postprocess_images(x_real).numpy()
            x_ref_post = postprocess_images(x_ref).numpy()

        x_real_stacked = x_real_post.transpose(1, 0, 2, 3).reshape(self.img_size, -1, 3)
        x_ref_stacked = x_ref_post.reshape(-1, self.img_size, 3)

        canvas.paste(PIL.Image.fromarray(np.uint8(x_real_stacked), 'RGB'), (self.img_size + 10, 0))
        canvas.paste(PIL.Image.fromarray(np.uint8(x_ref_stacked), 'RGB'), (0, self.img_size + 10))

        if self.strategy.num_replicas_in_sync > 1:
            x_ref = tf.concat(list(x_ref.values), axis=0)
            y_trg = tf.concat(list(y_trg.values), axis=0)

        for row in range(self.batch_size):
            row_images = tf.stack([x_ref[row]] * (self.batch_size // self.strategy.num_replicas_in_sync))
            row_images_y = tf.stack([y_trg[row]] * (self.batch_size // self.strategy.num_replicas_in_sync))

            x_fake = self.strategy.run(self.refer_canvas_inner, args=(x_real, row_images, row_images_y))

            if self.strategy.num_replicas_in_sync:
                x_fake = tf.concat(list(x_fake.values), axis=0)

            x_fake_stacked = postprocess_images(x_fake).numpy().transpose(1, 0, 2, 3).reshape(self.img_size, -1, 3)
            canvas.paste(PIL.Image.fromarray(np.uint8(x_fake_stacked), 'RGB'), (self.img_size + 10, (row + 1) * self.img_size + 10))

        return canvas

    @tf.function
    def refer_canvas_inner(self, x_real, x_ref, y_trg):
        s_trg = self.style_encoder_ema([x_ref, y_trg])
        x_fake = self.generator_ema([x_real, s_trg])
        return x_fake

    def latent_canvas(self, x_real, z_trg):
        canvas = PIL.Image.new('RGB', (self.img_size * (self.num_domains + 1) + 10, self.img_size * self.batch_size), 'white')

        if self.strategy.num_replicas_in_sync > 1:
            src_image = postprocess_images(tf.concat(list(x_real.values), axis=0)).numpy()
        else:
            src_image = postprocess_images(src_image).numpy()

        src_image_stacked = src_image.reshape(-1, self.img_size, 3)
        canvas.paste(PIL.Image.fromarray(np.uint8(src_image_stacked), 'RGB'), (0, 0))

        for col in range(self.num_domains):
            y_trg = tf.reshape(tf.constant([col]), shape=[1, 1])
            x_fake = self.strategy.run(self.latent_canvas_inner, args=(x_real, y_trg, z_trg))

            if self.strategy.num_replicas_in_sync > 1:
                x_fake = tf.concat(list(x_fake.values), axis=0)

            x_fake_stacked = postprocess_images(x_fake).numpy().reshape(-1, self.img_size, 3)
            canvas.paste(PIL.Image.fromarray(np.uint8(x_fake_stacked), 'RGB'), ((col + 1) * self.img_size + 10, 0))

        return canvas

    @tf.function
    def latent_canvas_inner(self, x_real, y_trg, z_trg):
        s_trg = self.mapping_network_ema([z_trg, y_trg])
        x_fake = self.generator_ema([x_real, s_trg])
        return x_fake

    def test(self, merge=True, merge_size=0):
        source_path = os.path.join(self.test_dataset_path, 'src_imgs')
        source_images = glob(os.path.join(source_path, '*.png')) + glob(os.path.join(source_path, '*.jpg'))
        source_images = sorted(source_images)

        # reference-guided synthesis
        print('reference-guided synthesis')
        reference_path = os.path.join(self.test_dataset_path, 'ref_imgs')
        reference_images = []
        reference_domain = []

        for idx, domain in enumerate(self.domain_list):
            image_list = glob(os.path.join(reference_path, domain) + '/*.png') + glob(
                os.path.join(reference_path, domain) + '/*.jpg')
            image_list = sorted(image_list)
            domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

            reference_images.extend(image_list)
            reference_domain.extend(domain_list)

        if merge:
            src_img = None
            ref_img = None
            ref_img_domain = None

            if merge_size == 0:
                # [len_src_imgs : len_ref_imgs] matching
                for src_idx, src_img_path in enumerate(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img_ = tf.expand_dims(src_img_, axis=0)

                    if src_idx == 0:
                        src_img = src_img_
                    else:
                        src_img = tf.concat([src_img, src_img_], axis=0)

                for ref_idx, (ref_img_path, ref_img_domain_) in enumerate(zip(reference_images, reference_domain)):
                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                    ref_name = os.path.basename(ref_name)

                    ref_img_ = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    ref_img_ = tf.expand_dims(ref_img_, axis=0)
                    ref_img_domain_ = tf.expand_dims(ref_img_domain_, axis=0)

                    if ref_idx == 0:
                        ref_img = ref_img_
                        ref_img_domain = ref_img_domain_
                    else:
                        ref_img = tf.concat([ref_img, ref_img_], axis=0)
                        ref_img_domain = tf.concat([ref_img_domain, ref_img_domain_], axis=0)

                save_path = './{}/ref_all.jpg'.format(self.result_dir)

                self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                  img_num=[len(source_images), len(reference_images)])

            else:
                # [merge_size : merge_size] matching
                src_size = 0
                for src_idx, src_img_path in enumerate(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img_ = tf.expand_dims(src_img_, axis=0)

                    if src_size < merge_size:
                        if src_idx % merge_size == 0:
                            src_img = src_img_
                        else:
                            src_img = tf.concat([src_img, src_img_], axis=0)
                        src_size += 1

                        if src_size == merge_size:
                            src_size = 0

                            ref_size = 0
                            for ref_idx, (ref_img_path, ref_img_domain_) in enumerate(
                                    zip(reference_images, reference_domain)):
                                ref_name, ref_extension = os.path.splitext(ref_img_path)
                                ref_name = os.path.basename(ref_name)

                                ref_img_ = load_images(ref_img_path, self.img_size,
                                                       self.img_ch)  # [img_size, img_size, img_ch]
                                ref_img_ = tf.expand_dims(ref_img_, axis=0)
                                ref_img_domain_ = tf.expand_dims(ref_img_domain_, axis=0)

                                if ref_size < merge_size:
                                    if ref_idx % merge_size == 0:
                                        ref_img = ref_img_
                                        ref_img_domain = ref_img_domain_
                                    else:
                                        ref_img = tf.concat([ref_img, ref_img_], axis=0)
                                        ref_img_domain = tf.concat([ref_img_domain, ref_img_domain_], axis=0)

                                    ref_size += 1
                                    if ref_size == merge_size:
                                        ref_size = 0

                                        save_path = './{}/ref_{}_{}.jpg'.format(self.result_dir, src_idx + 1, ref_idx + 1)

                                        self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                                          img_num=merge_size)

        else:
            # [1:1] matching
            for src_img_path in tqdm(source_images):
                src_name, src_extension = os.path.splitext(src_img_path)
                src_name = os.path.basename(src_name)

                src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                src_img = tf.expand_dims(src_img, axis=0)

                for ref_img_path, ref_img_domain in zip(reference_images, reference_domain):
                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                    ref_name = os.path.basename(ref_name)

                    ref_img = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    ref_img = tf.expand_dims(ref_img, axis=0)
                    ref_img_domain = tf.expand_dims(ref_img_domain, axis=0)

                    save_path = './{}/ref_{}_{}{}'.format(self.result_dir, src_name, ref_name, src_extension)

                    self.refer_canvas(src_img, ref_img, ref_img_domain, save_path, img_num=1)

        # latent-guided synthesis
        print('latent-guided synthesis')
        for src_img_path in tqdm(source_images):
            src_name, src_extension = os.path.splitext(src_img_path)
            src_name = os.path.basename(src_name)

            src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
            src_img = tf.expand_dims(src_img, axis=0)

            save_path = './{}/latent_{}{}'.format(self.result_dir, src_name, src_extension)

            self.latent_canvas(src_img, save_path)
