#
# A wrapper script that trains the ensemble SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#
# The first ensemble is baseline and pseudoresnet34

import os
import sys
import numpy as np
import cls_feature_class
import cls_data_generator
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
import keras_model
import parameter
import time
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras.models import load_model

global history 

import tensorflow
import math

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
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
      
    return sed, accdoa_in

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

        #Load the wanted models
        
        model1 = model2 = model3 = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], doa_objective=params['doa_objective'], is_accdoa=params['is_accdoa'],
                                      model_approach=0,
                                      depth = params['nb_conf'],
                                      decoder = params['decoder'],
                                      dconv_kernel_size = params['dconv_kernel_size'],
                                      nb_conf = params['nb_conf'])
        
        print('hey1')
        model1.load_weights(params['model_dir']+'2_lowfreq_base_da1_mic_dev_split6_model.h5')
        print('done')

        """
        model2 = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], doa_objective=params['doa_objective'], is_accdoa=params['is_accdoa'],
                                      model_approach=2,
                                      depth = params['nb_conf'],
                                      decoder = 0,
                                      dconv_kernel_size = params['dconv_kernel_size'],
                                      nb_conf = params['nb_conf'])
        #model2.load_weights(params['model_dir']+'2_resnet32_bs4_pseudoresnet_gpu_mic_dev_split6_model.h5'.format(unique_name))
        print("hey")
        """
        model2.load_weights(params['model_dir']+'2_midfreq_base_da1_mic_dev_split6_model.h5')
        model3.load_weights(params['model_dir']+'2_topfreq_base_da1_mic_dev_split6_model.h5')
    for epoch_cnt in range(nb_epoch):
        start = time.time()
        
        # predict once per epoch
        pred1 = model1.predict_generator(
            generator=data_gen_val.generate(),
            steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
            verbose=2
        )
        pred2 = model2.predict_generator(
            generator=data_gen_val.generate(),
            steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
            verbose=2
        )
        pred3 = model3.predict_generator(
            generator=data_gen_val.generate(),
            steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
            verbose=2
        )
        pred = (pred1 + pred2 +pred3)/3
        if params['is_accdoa']:
            sed_pred, doa_pred = get_accdoa_labels(pred, nb_classes)
            sed_pred= reshape_3Dto2D(sed_pred)
            doa_pred = reshape_3Dto2D(doa_pred)
        else:
            sed_pred = reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = reshape_3Dto2D(pred[1] if params['doa_objective'] is 'mse' else pred[1][:, :, nb_classes:])

        dcase_output_val_folder = os.path.join(params['dcase_output_dir'] if len(argv) < 3 else params['dcase_output_dir'] + argv[1] + '_' + argv[2], '{}_{}_{}_val'.format(task_id, params['dataset'], params['mode']))
        # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
        dump_DCASE2021_results(data_gen_val, feat_cls, dcase_output_val_folder, sed_pred, doa_pred)
        seld_metric[epoch_cnt, :] = score_obj.get_SELD_Results(dcase_output_val_folder)

        patience_cnt += 1
        if seld_metric[epoch_cnt, -1] < best_seld_metric:
            print('saving...')
            best_seld_metric = seld_metric[epoch_cnt, -1]
            best_epoch = epoch_cnt
            #model.save(model_name)
            print('saved!')
            patience_cnt = 0

        print(
            'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, '
            '\n\t\t DCASE2021 SCORES: ER: {:0.2f}, F: {:0.1f}, LE: {:0.1f}, LR:{:0.1f}, seld_score (early stopping score): {:0.2f}, '
            'best_seld_score: {:0.2f}, best_epoch : {}\n'.format(
                epoch_cnt, time.time() - start, tr_loss[epoch_cnt],
                seld_metric[epoch_cnt, 0], seld_metric[epoch_cnt, 1]*100,
                seld_metric[epoch_cnt, 2], seld_metric[epoch_cnt, 3]*100,
                seld_metric[epoch_cnt, -1], best_seld_metric, best_epoch
            )
        )
        if patience_cnt > params['patience']:
            break
        
        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tSELD_score (early stopping score) : {}'.format(best_seld_metric))

        print('\n\tDCASE2021 scores')
        print('\tClass-aware localization scores: Localization Error: {:0.1f}, Localization Recall: {:0.1f}'.format(seld_metric[best_epoch, 2], seld_metric[best_epoch, 3]*100))
        print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(seld_metric[best_epoch, 0], seld_metric[best_epoch, 1]*100))

        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('\nLoading the best model and predicting results on the testing split')
        print('\tLoading testing dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=split, shuffle=False, per_file=True, is_eval=True if params['mode'] is 'eval' else False
        )
        
        print(unique_name)
        #model = keras_model.load_seld_model('{}_model.h5'.format(unique_name), params['doa_objective'],params['model_approach'], False, '')
        pred_test1 = model1.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )
        pred_test2 = model2.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )
        pred_test3 = model3.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )
        pred_test = (pred_test1+pred_test2 + pred_test3)/3
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
    plt.show()
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)