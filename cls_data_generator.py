#
# Data generator for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random
from data_augmentation import *
from utils import *


class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False, train=False, tta = 0
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._is_accdoa = params['is_accdoa']
        self._doa_objective = params['doa_objective']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = self._feat_cls.get_nb_classes()
        self._get_filenames_list_and_feat_label_sizes()

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None

        #CUSTOM
        self.augm_indx = np.array([0], dtype=np.int8)
        self.data_augm = params['data_augm']
        self.augm_filenames = []
        #check if data is for training, if true then data augmentation can be used
        #else if it is for predict do not augment
        self.train = train
        self.model_approach = params['model_approach']
        self.tta = tta

        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._feature_batch_seq_len))))

        # self._dummy_feat_vec = np.ones(self._feat_len.shape) *

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._is_accdoa:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3)
            else:
                label_shape = [
                    (self._batch_size, self._label_seq_len, self._nb_classes),
                    (self._batch_size, self._label_seq_len, self._nb_classes*3) 
                ]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        
        for filename in os.listdir(self._feat_dir):
            if self._is_eval:
                self._filenames_list.append(filename)
            else:
                if int(filename[4]) in self._splits: # check which split the file belongs to
                    self._filenames_list.append(filename)

        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            self._label_len = temp_label.shape[-1]
            self._doa_len = (self._label_len - self._nb_classes)//self._nb_classes

        if self._per_file:
            self._batch_size = int(np.ceil(temp_feat.shape[0]/float(self._feature_seq_len)))

        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """

        while 1:
            if self._shuffle:
                random.shuffle(self._filenames_list)

            # Ideally this should have been outside the while loop. But while generating the test data we want the data
            # to be the same exactly for all epoch's hence we keep it here.
            self._circ_buf_feat = deque()
            self._circ_buf_label = deque()

            file_cnt = 0
            if self._is_eval:
                for i in range(self._nb_total_batches):
                    # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                    # circular buffer. If not keep refilling it.
                    while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                        for row_cnt, row in enumerate(temp_feat):
                            self._circ_buf_feat.append(row)

                        # If self._per_file is True, this returns the sequences belonging to a single audio recording
                        if self._per_file:
                            extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                            for row_cnt, row in enumerate(extra_feat):
                                self._circ_buf_feat.append(row)

                        file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                    feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                    for j in range(self._feature_batch_seq_len):
                        feat[j, :] = self._circ_buf_feat.popleft()
                    feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins)).transpose((0, 2, 1))

                    # Split to sequences
                    feat = self._split_in_seqs(feat, self._feature_seq_len)
                    feat = np.transpose(feat, (0, 3, 1, 2))

                    yield feat

            else:
                for i in range(self._nb_total_batches):

                    # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                    # circular buffer. If not keep refilling it.
                    while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                        #CUSTOM augment file by file
                        #(3000, 640)->(3000, 10, 64)->(10, 300, 10, 64)
                        '''
                        temp_f_row = np.reshape(f_row, (temp_f_row.shape[0], self._nb_ch, self._nb_mel_bins)).transpose((0, 2, 1))
                        temp_f_row = self._split_in_seqs(temp_f_row, self._feature_seq_len)
                        temp_f_row = np.transpose(temp_f_row, (0, 3, 1, 2))

                        ##CUSTOM add a data augmented feature in the array
                        #for each batch segment(64 segments) add that to specaugm
                        if self.data_augm == 1:
                            #reset index array 
                            self.augm_indx = np.array([0], dtype=np.int8)
                            for j in range(temp_f_row.shape[0]):
                                was_augm = False
                                feat_augm, was_augm = SpecAugmentNp(nb_ch = self._nb_ch, nb_mel_bins=self._nb_mel_bins)(temp_f_row[j,:,:,:])
                                if was_augm is True:
                                    #have a list of the indexes of files that were augmented, to copy the labels on that index
                                    #np.insert(feat, j, [feat_augm], axis=0)
                                    for f_row in feat_augm:
                                        self._circ_buf_feat.append(f_row)
                                    self.augm_indx = np.append(self.augm_indx, int(j))
                                    self.augm_filenames = np.append(self.augm_filenames, self._filenames_list[file_cnt])
                        '''
                        for f_row in temp_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in temp_label:
                            self._circ_buf_label.append(l_row)

                        # If self._per_file is True, this returns the sequences belonging to a single audio recording
                        if self._per_file:
                            feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                            label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                            for f_row in extra_feat:
                                self._circ_buf_feat.append(f_row)
                            for l_row in extra_labels:
                                self._circ_buf_label.append(l_row)

                        file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                    feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                    label = np.zeros((self._label_batch_seq_len, self._label_len))

                    for j in range(self._feature_batch_seq_len):
                        #(19200, 640) = (feature_batch_len, ch*mel_bins)
                        feat[j, :] = self._circ_buf_feat.popleft()
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()

                    #(19200, 10, 64) (BATCH+SEQ_LEN, CHAN, MEL_BINS)
                    feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins)).transpose((0, 2, 1))

                    # Split to sequences
                    #(64, 300, 64, 10) (batches, SEQ_LEN, MEL_BINS, CHAN)
                    feat = self._split_in_seqs(feat, self._feature_seq_len)
                    feat = np.transpose(feat, (0, 3, 1, 2))
                    
                    ##CUSTOM if ensemble method of 3 freq is used 
                    #then generate 3 feat instead of  and 3 respective labels
                    if self.model_approach == 4:
                        feat1 = EnsembleFreqMasking(min = 0, max = 32)(feat)
                        feat2 = EnsembleFreqMasking(min = 16, max = 48)(feat)
                        feat3 = EnsembleFreqMasking(min = 32, max = 64)(feat)

                        feat = np.concatenate(feat1, feat2, feat3)

                    
                    ##CUSTOM add a data augmented feature in the array
                    #for each batch segment(64 segments) add that to specaugm
                    if self.data_augm > 0 and self.train:
                        temp_feat_full = np.zeros((4*feat.shape[0],self._nb_ch, self._feature_seq_len,self._nb_mel_bins))
                        temp_label_full = np.zeros((4*feat.shape[0], self._label_seq_len, self._label_len))
                        #reset index array 
                        self.augm_indx = np.array([0], dtype=np.int8)
                        counter = 0
                        for j in range(feat.shape[0]):
                            temp_feat_full[j,:,:,:] = feat[j,:,:,:]
                            was_augm = False
                            if self.data_augm == 1:
                                feat_augm, was_augm = SpecAugmentNp()(feat[j, :, :, :])
                            elif self.data_augm == 2:
                                feat_augm, was_augm = RandomShiftUpDownNp(freq_shift_range=10)(feat[j, :, :, :])
                            
                            if was_augm is True:
                                counter +=1
                                #have a list of the indexes of files that were augmented, to copy the labels on that index
                                temp_feat_full[feat.shape[0]+counter-1,:,:,:] = feat_augm
                                #np.insert(feat, j, [feat_augm], axis=0)
                                self.augm_indx = np.append(self.augm_indx, int(j))
                        
                        feat = temp_feat_full[:feat.shape[0]+counter,:,:,:]
                    
                    #(64, 60, 48) (BATCH SIZE, SEQ LEN, CLASSES)
                    label = self._split_in_seqs(label, self._label_seq_len)
                   
                    ##CUSTOM if ensemble method of 3 freq is used 
                    #then generate 3 feat instead of  and 3 respective labels
                    if self.model_approach == 4:
                        label = np.concatenate(label, label, label)
                    if self.data_augm is not 0 and self.train:
                        counter = 0
                        for j in range(label.shape[0]):
                            temp_label_full[j,:,:] = label[j,:,:]
                            if self.data_augm < 4:
                                for i in range(1,len(self.augm_indx)):
                                    #copy the label of the previous step
                                    if self.augm_indx[i] == j :
                                        counter+=1
                                        temp_label_full[label.shape[0]+counter-1,:,:] = label[j,:,:]
                                        #np.insert(label, j, [label[j,:,:]], axis=0)
                                        break
                        if self.data_augm == 4:
                            counter = 0
                            for j in range(feat.shape[0]):
                                feat_augm = []
                                label_seds = []
                                label_doas = []
                                #if self.tta == 1:
                                feat_news, label_seds, label_doas, was_augm = GccSwapChannelMic( tta = 1)(x = feat[j,:,:,:], y_sed= label[j,:,:self._nb_classes], y_doa= label[j,:,self._nb_classes:])
                                if was_augm is True:
                                    counter +=1
                                    temp_feat_full[feat.shape[0]+counter-1,:,:,:] = feat_news
                                    temp_label_full[feat.shape[0]+counter-1,:,:self._nb_classes] = label_seds
                                    temp_label_full[feat.shape[0]+counter-1,:,self._nb_classes:] = label_doas
                                #elif self.tta == 2:
                                feat_news, label_seds, label_doas, was_augm = GccSwapChannelMic( tta = 2)(x = feat[j,:,:,:], y_sed= label[j,:,:self._nb_classes], y_doa= label[j,:,self._nb_classes:])
                                if was_augm is True:
                                    counter +=1
                                    temp_feat_full[feat.shape[0]+counter-1,:,:,:] = feat_news
                                    temp_label_full[feat.shape[0]+counter-1,:,:self._nb_classes] = label_seds
                                    temp_label_full[feat.shape[0]+counter-1,:,self._nb_classes:] = label_doas
                                #elif self.tta == 3:
                                feat_news, label_seds, label_doas, was_augm = GccSwapChannelMic( tta = 3)(x = feat[j,:,:,:], y_sed= label[j,:,:self._nb_classes], y_doa= label[j,:,self._nb_classes:])
                                if was_augm is True:
                                    counter +=1
                                    temp_feat_full[feat.shape[0]+counter-1,:,:,:] = feat_news
                                    temp_label_full[feat.shape[0]+counter-1,:,:self._nb_classes] = label_seds
                                    temp_label_full[feat.shape[0]+counter-1,:,self._nb_classes:] = label_doas
                                feat_news, label_seds, label_doas, was_augm = GccSwapChannelMic( tta = 4)(x = feat[j,:,:,:], y_sed= label[j,:,:self._nb_classes], y_doa= label[j,:,self._nb_classes:])
                                
                                if was_augm is True:
                                    counter +=1
                                    temp_feat_full[feat.shape[0]+counter-1,:,:,:] = feat_news
                                    temp_label_full[feat.shape[0]+counter-1,:,:self._nb_classes] = label_seds
                                    temp_label_full[feat.shape[0]+counter-1,:,self._nb_classes:] = label_doas
                            feat = temp_feat_full[:feat.shape[0]+counter,:,:,:]
                        label = temp_label_full[:label.shape[0]+counter,:,:]
                        
                    #17/5/2022 CUSTOM add temp variable to keep the y_Sed->needed for tta
                    y_sed = label[:, :, :self._nb_classes]
                    if self._is_accdoa:
                        mask = label[:, :, :self._nb_classes]
                        mask = np.tile(mask, 3)
                        label = mask * label[:, :, self._nb_classes:]

                    else:
                         label = [
                            label[:, :, :self._nb_classes],  # SED labels
                            label[:, :, self._nb_classes:] if self._doa_objective is 'mse' else label # SED + DOA labels
                             ]

                    #14/5/2022 CUSTOM for each batch of feat and labels, do tta with ACS (GccRandomSwapMic)
                    #(10, 60, 36)=(B, T, F) label size
                    #(10, 10, 60, 36)=(B, C, T, F) feat size
                    if self.tta > 0:
                        #np.zeros((2*feat.shape[0],self._nb_ch, self._feature_seq_len,self._nb_mel_bins))
                        feat_news = np.zeros((label.shape[0], self._nb_ch, self._feature_seq_len,self._nb_mel_bins))
                        label_seds = np.zeros((label.shape[0], label.shape[1], self._nb_classes))
                        label_doas = np.zeros((label.shape[0], label.shape[1], label.shape[2]))

                        for i in range(label.shape[0]):
                            if self.tta == 1:
                                feat_news[i,:,:,:], label_seds[i,:,:], label_doas[i,:,:], was_augm = GccSwapChannelMic(always_apply=True, tta = self.tta)(x = feat[i,:,:,:], y_sed= y_sed[i,:,:], y_doa= label[i,:,:])
                            elif self.tta == 2:
                                feat_news[i,:,:,:], label_seds[i,:,:], label_doas[i,:,:], was_augm = GccSwapChannelMic(always_apply=True, tta = self.tta)(x = feat[i,:,:,:], y_sed= y_sed[i,:,:], y_doa= label[i,:,:])
                            elif self.tta == 3:
                                feat_news[i,:,:,:], label_seds[i,:,:], label_doas[i,:,:], was_augm = GccSwapChannelMic(always_apply=True, tta = self.tta)(x = feat[i,:,:,:], y_sed= y_sed[i,:,:], y_doa= label[i,:,:])
                        
                        feat, label = feat_news, label_doas

                    yield feat, label

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_default_elevation(self):
        return self._default_ele

    def get_azi_ele_list(self):
        return self._feat_cls.get_azi_ele_list()

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_classes(self):
        return self._feat_cls.get_classes()
    
    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)