config = {'datapath':'/home/ubuntu/DSB2017/luna/raw/',
          'segment_datapath':'/home/ubuntu//DSB2017/luna/segment/',
          'outputfile':'prediction.csv',
          
          'luna_raw':'./luna/raw/',
          'luna_segment':'./luna/segment/',
          'luna_data':'./luna/allset',
          'preprocess_result_path':'/home/ubuntu/DSB2017/preprocess/',      
          'luna_abbr':'/home/ubuntu/DSB2017/training/detector/labels/shorter.csv',

          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/classifier.ckpt',
         'n_gpu':8,
         'n_worker_preprocessing':None,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':False,
         'skip_detect':False}
