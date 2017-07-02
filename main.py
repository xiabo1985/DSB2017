from preprocessing import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas

segmentdatapath = config_submit['segment_datapath']
datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if not skip_prep:
    #testsplit对应的是病人id的集合，每个id都有一个id_label.npy的文件存储了该病人的每个结节对应的lable信息的集合！！此处主要为了告知lable的存储文件的。
    testsplit = full_prep(segmentdatapath,datapath, prep_result_path,
                          n_worker=config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
else:
    testsplit = os.listdir(datapath)

nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
#testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    margin = 32
    sidelen = 144
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)

    test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])




