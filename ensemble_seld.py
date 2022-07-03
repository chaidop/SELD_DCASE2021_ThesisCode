#
# A wrapper script that trains the ensemble SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#
# The first ensemble is baseline and pseudoresnet34

from operator import mod
import os
import sys
import numpy as np
from tensorboard import summary
import cls_feature_class
import cls_data_generator
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
import keras_model
import parameter
import parameter2 # for 128 mel
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import keras
from keras.models import load_model, Model
import swa

global history 

import tensorflow
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sd=[]
class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

history=LossHistory()
import tensorflow

def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model
def get_model(params, data_in, data_out, app, decoder):
    import keras
    keras.backend.set_image_data_format('channels_first')
    model = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], doa_objective=params['doa_objective'], is_accdoa=params['is_accdoa'],
                                      model_approach=app,
                                      depth = params['nb_conf'],
                                      decoder = decoder,
                                      dconv_kernel_size = params['dconv_kernel_size'],
                                      nb_conf = params['nb_conf'],
                                      simple_parallel=params['simple_parallel'])
    return model

    
# learning rate schedule
def step_decay(losses):
    print('decay ')
    if float(2*np.sqrt(np.array(abs(history.losses[-1]))))<0.3:
        print('changed')
        lrate=0.0001*1/(1+0.01*len(history.losses))
        momentum=0.8
        decay_rate=2e-6
        return lrate
    else:
        print('hist loss ',history.losses[-1])
        lrate=0.0001
        return lrate
lrate= tensorflow.keras.callbacks.LearningRateScheduler(step_decay)

def rotate_back(y_doa, tta, n_classes):
    y_doa_new = y_doa.copy()
    # change y_doa
    if y_doa.shape[-1] == 3 * n_classes:  # classwise reg_xyz, accdoa
        if tta == 1: # swap M2 and M3 -> swap x and y
            y_doa_new[:, :, 0:n_classes] = y_doa[:,:, n_classes:2*n_classes]
            y_doa_new[:,:, n_classes:2*n_classes] = y_doa[:,:, :n_classes]
        elif tta == 2:  # swap M1 and M4 -> swap x and y, negate x and y
            temp = - y_doa_new[:,:, 0:n_classes].copy()
            y_doa_new[:,:, 0:n_classes] = - y_doa_new[:,:, n_classes:2 * n_classes]
            y_doa_new[:,:, n_classes:2 * n_classes] = temp
        elif tta == 3:  # swap M1 and M2, M3 and M4 -> negate y and z
            y_doa_new[:,:, n_classes:2 * n_classes] = - y_doa_new[:,:, n_classes:2 * n_classes]
            y_doa_new[:,:, 2 * n_classes:] = - y_doa_new[:,:, 2 * n_classes:]
        elif tta == 4:  # swap M1 and M2, M2 and M4, M3 and M1, M4 and M3 -> swap x and y, negate y and z
                y_doa_new[:, 0:n_classes] = y_doa[:, n_classes:2*n_classes]
                y_doa_new[:, n_classes:2*n_classes] = -y_doa[:, :n_classes]
                y_doa_new[:, n_classes:2 * n_classes] = - y_doa_new[:, n_classes:2 * n_classes]
                y_doa_new[:, 2 * n_classes:] = - y_doa_new[:, 2 * n_classes:]

        elif tta == 5:  # swap M1 and M3, M2 and M1, M3 and M4, M4 and M2 -> swap x and y, negate x and z
            y_doa_new[:, n_classes:2 * n_classes] = - y_doa[:,0:n_classes]
            y_doa_new[:, 0:n_classes] = y_doa[:, n_classes:2 * n_classes]
            y_doa_new[:, 0:n_classes] = - y_doa_new[:, 0:n_classes]
            y_doa_new[:, 2 * n_classes:] = - y_doa_new[:, 2 * n_classes:]

        elif tta == 6:  # swap M1 and M4 and M2 and M3 -> swap x and y, negate x and y
            y_doa_new[:, n_classes:2 * n_classes] = - y_doa[:,0:n_classes]
            y_doa_new[:, 0:n_classes] = y_doa[:, n_classes:2 * n_classes]
            y_doa_new[:, 0:n_classes] = - y_doa_new[:, 0:n_classes]
            y_doa_new[:, n_classes:2 * n_classes] = - y_doa_new[:, n_classes:2 * n_classes]

        elif tta == 7:  # swap M1 and M3, M2 and M4, M3 -> negate x and z
            y_doa_new[:, 0:n_classes] = - y_doa_new[:, 0:n_classes]
            y_doa_new[:, :, 2 * n_classes:] = - y_doa_new[:, :, 2 * n_classes:]
    
    return y_doa_new
def dump_DCASE2021_results(_data_gen, _feat_cls, _dcase_output_folder, _sed_pred, _doa_pred):
    '''
    Write the filewise results to individual csv files
    '''

    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    max_frames_with_content = _data_gen.get_nb_frames()

    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with
    # zero padding in the remaining frames
    test_filelist = _data_gen.get_filelist()
    frames_per_file = _data_gen.get_frame_per_file()
    for file_cnt in range(_sed_pred.shape[0] // frames_per_file):
        output_file = os.path.join(_dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
        dc = file_cnt * frames_per_file
        output_dict = _feat_cls.regression_label_format_to_output_format(
            _sed_pred[dc:dc + max_frames_with_content, :],
            _doa_pred[dc:dc + max_frames_with_content, :]
        )
        _data_gen.write_output_format_file(output_file, output_dict)
    return


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > [0.35, 0.36, 0.3, 0.4, 0.65, 0.6, 0.45, 0.55, 0.3, 0.3, 0.45, 0.3] #0.5
      
    return sed, accdoa_in

#Kos ensembling bDNN
@tf.function
def predict(model, x, batch_size, win_size, step_size):
    windows = tf.signal.frame(x, win_size, step_size, axis=0)

    accdoa = []
    for i in range(int(np.ceil(windows.shape[0]/batch_size))):
        ac = model(windows[i*batch_size:(i+1)*batch_size], training=False)
        accdoa.append(ac)
    accdoa = tf.concat(accdoa, axis=0)

    # windows to seq
    total_counts = tf.signal.overlap_and_add(
        tf.ones((accdoa.shape[0], win_size//step_size), dtype=accdoa.dtype),
        1)[..., tf.newaxis]
    accdoa = tf.signal.overlap_and_add(tf.transpose(accdoa, (2, 0, 1)), 1)
    accdoa = tf.transpose(accdoa, (1, 0)) / total_counts

    return accdoa

def ensemble_outputs(model, xs: list,
                     win_size=300, step_size=5, batch_size=4):
    # assume 0th dim of each sample is time dim
    accdoas = []

    for x in xs:
        accdoa = predict(model, x, batch_size, win_size, step_size)
        accdoas.append(accdoa)

    return list(accdoa)
def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(tf.test.is_built_with_cuda())
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')
    
    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameter.get_params(task_id)
    params2 = parameter2.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    feat_cls = cls_feature_class.FeatureClass(params)
    train_splits, val_splits, test_splits = None, None, None

    if params['mode'] == 'dev':
        test_splits = [6]
        val_splits = [5]
        train_splits = [[1, 2, 3, 4]]

    elif params['mode'] == 'eval':
        test_splits = [[7, 8]]
        val_splits = [[6]]
        train_splits = [[1, 2, 3, 4, 5]]

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_{}_split{}'.format(
            task_id, job_id, params['dataset'], params['mode'], split
        )
        unique_name = os.path.join(params['model_dir'], unique_name)
        model_name = '{}_model.h5'.format(unique_name)
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )
        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True, is_eval=False
        )

        # Collect the reference labels for validation data
        data_in, data_out = data_gen_train.get_data_sizes()
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))

        nb_classes = data_gen_train.get_nb_classes()
        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n\tdoa_objective: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size'], params['doa_objective']))

        
        print('Using loss weights : {}'.format(params['loss_weights']))

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        best_seld_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss = np.zeros(nb_epoch)
        seld_metric = np.zeros((nb_epoch, 5))
                
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=split, shuffle=False, per_file=True, is_eval=True if params['mode'] is 'eval' else False
        )
        #Load the wanted models
        model1 = get_model(params, data_in, data_out, 13, 0)
        #model1.load_weights(params['model_dir']+'2_conformer_swa_da1_2_3_tta2_dyn_mic_dev_split6_model.h5')
        
        model2 = get_model(params, data_in, data_out, 12, 0)
        '''
        keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], doa_objective=params['doa_objective'], is_accdoa=params['is_accdoa'],
                                      model_approach=7,
                                      depth = params['nb_conf'],
                                      decoder = 0,
                                      dconv_kernel_size = params['dconv_kernel_size'],
                                      nb_conf = params['nb_conf'],
                                      simple_parallel=params['simple_parallel'])
'''

        print('model2 done')
        #model1.load_weights(params['model_dir']+'2_swa_baseline_da2_mic_dev_split6_model.h5')
        model1.load_weights('models/2_densenet_da1_2_3_tta_dyn_mic_dev_split6_model.h5')
        print('model1 done')
        #model2.load_weights(params['model_dir']+'2_conformer_swa_da1_2_3_tta2_dyn_mic_dev_split6_model.h5')
        #squeeze-conformer
        model2.load_weights(params['model_dir']+'2_conformer_da1_2_3_tta4_no2dense_mic_dev_split6_model.h5')
        print('weights2 done')
        
        model7 = get_model(params, data_in, data_out, 10, 0)
        print('model7 done')
        model7.load_weights(params['model_dir']+'2_new_resnet34_conforer_tta2_da1_2_3_acs1_dyn_mic_dev_split6_model.h5')
        print('weights3 done')
        #model7.summary()
        print('model3 done', unique_name)
        #model3.save(params['model_dir']+'2_ready_conformer_depth_swa_myda2_adam0_0_0001_nogru_scheduler_wreg_extradenselayer_noinverse_256_mic_dev_split6_model.h5')
        model4 = get_model(params, data_in, data_out, 6, 0)
        print('model3 done')
        model4.load_weights(params['model_dir']+'2_conformer_swa_da1_2_3_tta2_dyn_mic_dev_split6_model.h5')
        print('weights4 done')
        model3 = get_model(params, data_in, data_out, 2, 0)
        #model3.load_weights(params['model_dir']+'2_resnet34_tta2_acs_da1_2_mic_dev_split6_model.h5')
        #resnet34
        model3.load_weights(params['model_dir']+'2_new_resnet34_tta2_da2_lenovo_mic_dev_split6_model.h5')
        print('weights3 done')

        model5 = get_model(params, data_in, data_out, 10, 0)
        #resnet-conformer
        model5.load_weights(params['model_dir']+ '2_new_resnet34_conforer_tta2_da1_2_3_acs1_dyn_mic_dev_split6_model.h5')
        
        model6 = get_model(params, data_in, data_out, 8, 0)

        model6.load_weights(params['model_dir']+'2_densenet_conforemr_da1_2_3_tta_dyn_mic_dev_split6_model.h5')

        
        model8 = get_model(params, data_in, data_out, 7, 0)

        model8.load_weights(params['model_dir']+'2_resnet2020_conformer_tta2_acs_mic_dev_split6_model.h5')

        ##ensembling
        models = [model1, model8, model5]

        from keras.layers import Input, Average
        from keras.models import Model
        model_input = Input(shape=( data_in[-3], data_in[-2], data_in[-1]))
        print(model_input)
        model_outputs = [model(model_input) for model in models] #list of models outputs
        ensemble_output = Average()(model_outputs)
        ensemble_model = Model(inputs=model_input, outputs=ensemble_output)


        #model = keras_model.load_seld_model('{}_model.h5'.format(unique_name), params['doa_objective'],params['model_approach'], False, '')
        pred_test = ensemble_model.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )
        #Kos aggregation
        if params['aggregation'] == True:
            outs = []
            for model in models:
                outs.append(ensemble_outputs(model, data_gen_test, batch_size=params['batch_size']))
                #del model
            
            outs1 = [outs[0]]
            outputs = []
            outs2 = list(zip(*outs1))
            for out in outs2:
                accdoa = list(zip(*out))
                accdoa = tf.add_n(accdoa) / len(accdoa)
                outputs.append((accdoa))

            print(outputs)
        #####
        
        # stochastic weight averaging
            '''swa_start_epoch = 10
            swa_freq = 2
            swa_obj = swa.SWA(model1, swa_start_epoch, swa_freq)
            swa_obj2 = swa.SWA(model2, swa_start_epoch, swa_freq)
            swa_obj3 = swa.SWA(model3, swa_start_epoch, swa_freq)
            swa_obj4 = swa.SWA(model4, swa_start_epoch, swa_freq)

            swa_obj.on_train_end()
            swa_obj2.on_train_end()
            swa_obj3.on_train_end()
            swa_obj4.on_train_end()'''
        # TTA
        if params['tta'] is True:
            print('tta')
            predictions = []
            data_gen_test_tta1 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 1, is_eval=True if params['mode'] is 'eval' else False
            )
            data_gen_test_tta2 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 2, is_eval=True if params['mode'] is 'eval' else False
            )
            '''data_gen_test_tta3 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 3, is_eval=True if params['mode'] is 'eval' else False
            )
            data_gen_test_tta4 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 4, is_eval=True if params['mode'] is 'eval' else False
            )
            data_gen_test_tta5 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 5, is_eval=True if params['mode'] is 'eval' else False
            )
            data_gen_test_tta6 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 6, is_eval=True if params['mode'] is 'eval' else False
            )
            data_gen_test_tta7 = cls_data_generator.DataGenerator(
                params=params, split=split, shuffle=False, per_file=True, train=False, tta = 7, is_eval=True if params['mode'] is 'eval' else False
            )'''
            pred_test_tta1 = ensemble_model.predict_generator(
                generator=data_gen_test_tta1.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            ) 
            pred_test_tta2 = ensemble_model.predict_generator(
                generator=data_gen_test_tta2.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            )
            '''pred_test_tta3 = ensemble_model.predict_generator(
                generator=data_gen_test_tta3.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            ) 
            pred_test_tta4 = ensemble_model.predict_generator(
                generator=data_gen_test_tta4.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            )
            pred_test_tta5 = ensemble_model.predict_generator(
                generator=data_gen_test_tta5.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            ) 
            pred_test_tta6 = ensemble_model.predict_generator(
                generator=data_gen_test_tta6.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            )
            pred_test_tta7 = ensemble_model.predict_generator(
                generator=data_gen_test_tta7.generate(),
                steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
                verbose=2
            ) '''
            
            print('pred tta', pred_test.shape)
            print('pred tta', pred_test_tta1.shape)
            print('pred tta', pred_test_tta2.shape)
            #rotate back
            pred_test_tta1 = rotate_back(pred_test_tta1, tta=1, n_classes=nb_classes) 
            pred_test_tta2 = rotate_back(pred_test_tta2, tta=2, n_classes=nb_classes)
            #pred_test_tta3 = rotate_back(pred_test_tta3, tta=3, n_classes=nb_classes) 
            #pred_test_tta4 = rotate_back(pred_test_tta4, tta=4, n_classes=nb_classes)
            #pred_test_tta5 = rotate_back(pred_test_tta5, tta=5, n_classes=nb_classes) 
            #pred_test_tta6 = rotate_back(pred_test_tta6, tta=6, n_classes=nb_classes)
            #pred_test_tta7 = rotate_back(pred_test_tta7, tta=7, n_classes=nb_classes) 

            predictions.append(pred_test)
            predictions.append(pred_test_tta1)
            predictions.append(pred_test_tta2)
            #predictions.append(pred_test_tta3)
            #predictions.append(pred_test_tta4)
            #predictions.append(pred_test_tta5)
            #predictions.append(pred_test_tta6)
            #predictions.append(pred_test_tta7)
        ##
            pred_test = np.mean(predictions, axis=0)
        if params['is_accdoa']:
            test_sed_pred, test_doa_pred = get_accdoa_labels(pred_test, nb_classes)
            test_sed_pred = reshape_3Dto2D(test_sed_pred)
            test_doa_pred = reshape_3Dto2D(test_doa_pred)
        else:
            test_sed_pred = reshape_3Dto2D(pred_test[0]) > 0.5
            test_doa_pred = reshape_3Dto2D(pred_test[1] if params['doa_objective'] is 'mse' else pred_test[1][:, :, nb_classes:])

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'] if len(argv) < 3 else params['dcase_output_dir'] + argv[1] + '_' + argv[2], '{}_{}_{}_test'.format(task_id, params['dataset'], params['mode']))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))
        dump_DCASE2021_results(data_gen_test, feat_cls, dcase_output_test_folder, test_sed_pred, test_doa_pred)

        if params['mode'] is 'dev':
            # Calculate DCASE2021 scores
            test_seld_metric = score_obj.get_SELD_Results(dcase_output_test_folder)

            print('Results on test split:')
            print('\tDCASE2021 Scores')
            print('\tClass-aware localization scores: Localization Error: {:0.1f}, Localization Recall: {:0.1f}'.format(test_seld_metric[2], test_seld_metric[3]*100))
            print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(test_seld_metric[0], test_seld_metric[1]*100))
            print('\tSELD (early stopping metric): {:0.2f}'.format(test_seld_metric[-1]))
    ensemble_model.save_weights(model_name)#modified because of memoryerror, it was save instead
    model_freezed = freeze_layers(ensemble_model)
    model_freezed.save(model_name + 'ensemble_model')
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)