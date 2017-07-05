config = {'outputfile':'prediction.csv',
          
          'luna_raw':'/home/ubuntu/DSB2017/luna/raw/',
          'luna_segment':'/home/ubuntu/DSB2017/luna/segment/seg-lungs-LUNA16',
          'luna_data':'/home/ubuntu/DSB2017/luna/allset',
          'preprocess_result_path':'/home/ubuntu/DSB2017/preprocess/',      
          'luna_abbr':'/home/ubuntu/DSB2017/training/detector/labels/shorter.csv',

          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/classifier.ckpt',
         'n_gpu':1,
         'n_worker_preprocessing':None,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':False,
         'skip_detect':False}
