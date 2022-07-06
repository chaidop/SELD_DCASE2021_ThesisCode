# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


from pickle import TRUE


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
         #####CUSTOM PARAMETERS ##############

        model_approach=12,  #####shows approach to be taken for seld (1 to run baseline)
        # 0 for baseline
        # 1 for resnet 18
        # 2 for resnet 34
        # 3 for conformer block
        # 4 for ensemble method of 3 diff freq, in order to detect overlaping events of same class
        # if on atleast 2 models there is the same class active, then it is possible we have same overlapping class
        # so average the predictions->might need to do SED and then DOA 
        # 6 ready conformer
        # 7 resnet_2020-conformer
        # 8 densenet
        # 9 CNN-squeeze
        # 10 resnet34-conformer
        # 12 squeeze-conformer


        dconv_kernel_size = 31, ## size of depthwise convolution for conformer approach
        nb_conf = 2,            ## number of conformer layers before SED and DOA separation 

        decoder = 0,
        # 0 for bi-gru
        # 1 for lstm

        data_augm = 0,
        # 0: None
        # 1: masking
        # 2: random shift up/down
        # 3: apply all da techniques
        # 4: acs

        learning_rate = 0.01,

        simple_parallel = False,

        tta = False,
        aggregation = False,
        agc = False,
        #####END CUSTOM PARAMETERS ############
        quick_test=False,           # If True: Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir='./input_data',  # Base folder containing the foa_dev/mic_dev and metadata folders

        # OUTPUT PATH
        feat_label_dir='./seld_feat_label2/',  # Directory to dump extracted features and labels
        model_dir='./models/',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='./results/',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',          # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,#original 64

        # DNN MODEL PARAMETERS
        is_accdoa=True,             # True: Use ACCDOA output format
        doa_objective='mse',        # if is_accdoa=True this is ignored, otherwise it supports: mse, masked_mse. where mse- original seld approach; masked_mse - dcase 2020 approach

        label_sequence_length=60,   # Feature sequence length
        batch_size= 64,             # Batch size ###############ORIGINAL 256
        dropout_rate=0.05,          # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer#original 64
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 1000.],   # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=7,               # Train for maximum epochs #### originally 50
        epochs_per_fit=5,           # Number of epochs per fit

        # METRIC PARAMETERS
        lad_doa_thresh=20
       
    )
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    params['unique_classes'] = {
            'alarm': 0,
            'baby': 1,
            'crash': 2,
            'dog': 3,
            'female_scream': 4,
            'female_speech': 5,
            'footsteps': 6,
            'knock': 7,
            'male_scream': 8,
            'male_speech': 9,
            'phone': 10,
            'piano': 11#, 'engine':12, 'fire':13
        }

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '3':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '4':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '5':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params