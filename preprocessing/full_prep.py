import os
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
from step1 import step1_python
import warnings
import SimpleITK as sitk
from config_submit import config as config_submit
import pandas
import shutil

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

# def savenpy(id):
id = 1

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')



def savenpy_luna(id ,filelist,luna_segment,luna_data,savepath,use_existing=True):
    print('start savenpy_luna...')
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    #name = name[:-4]
    #print ('name' + name)
    if use_existing:
        if os.path.exists(os.path.join(savepath, name + '_label.npy')) and os.path.exists(
                os.path.join(savepath, name + '_clean.npy')):
            print(name + ' had been done')
            return
    try:
        Mask, origin, spacing, isflip = load_itk_image(os.path.join(luna_segment, name + '.mhd'))
        if isflip:
            Mask = Mask[:, ::-1, ::-1]
        newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')
        m1 = Mask == 3
        m2 = Mask == 4
        Mask = m1 + m2

        xx, yy, zz = np.where(Mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack(
            [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T


        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name + '.mhd'))
        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        bones = (sliceim * extramask) > bone_thresh
        sliceim[bones] = pad_value

        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)
        np.save(os.path.join(savepath, name + '_label'), np.array([[0, 0, 0, 0]]))

    except:
        print('bug in ' + name)
        raise
    print(name + ' done')
    print('end savenpy_luna...')

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def prepare_luna(luna_segment,luna_raw):
    print('start changing luna name')
    luna_abbr = config_submit['luna_abbr']
    luna_data = config_submit['luna_data']
    finished_flag = '.flag_prepareluna'
    
    if not os.path.exists(finished_flag):
    
        subsetdirs = [os.path.join(luna_raw, f) for f in os.listdir(luna_raw) if
                      f.startswith('subset') and os.path.isdir(
                          os.path.join(luna_raw, f))]  
        print subsetdirs
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)

        abbrevs = np.array(
            pandas.read_csv(config_submit['luna_abbr'], header=None))  
        namelist = list(abbrevs[:, 1])
        ids = abbrevs[:, 0]
        print ([os.path.join(luna_raw, f) for f in os.listdir(luna_raw) if
                      f.startswith('subset') and os.path.isdir(
                          os.path.join(luna_raw, f))]  )
        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()
            for f in files:  
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0' * (3 - len(str(id))) + str(id)
                shutil.move(os.path.join(d, f), os.path.join(luna_data, filename + f[-4:]))
                print(os.path.join(luna_data, str(id) + f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data, file), 'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0' * (3 - len(str(id))) + str(id)
                content[-1] = 'ElementDataFile = ' + filename + '.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data, file), 'w') as f:
                f.writelines(content)

        seglist = os.listdir(luna_segment)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0' * (3 - len(str(id))) + str(id)

                shutil.move(os.path.join(luna_segment, f), os.path.join(luna_segment, filename + lastfix))
                print(os.path.join(luna_segment, filename + lastfix))

        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_segment, file), 'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0' * (3 - len(str(id))) + str(id)
                content[-1] = 'ElementDataFile = ' + filename + '.zraw\n'
                print(content[-1])
            with open(os.path.join(luna_segment, file), 'w') as f:
                f.writelines(content)
    print('end changing luna name')
    f= open(finished_flag,"w+")



def full_prep(luna_segment_path,luna_raw_path,prep_folder,n_worker = None,use_existing=True):
    prepare_luna(luna_segment_path,luna_raw_path)

    warnings.filterwarnings("ignore")

    luna_data = config_submit['luna_data']

    filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd')]

    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

        
    print('starting preprocessing')
    pool = Pool(n_worker)
    #partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,data_path=data_path,use_existing=use_existing)
    partial_savenpy = partial(savenpy_luna, filelist=filelist, luna_segment=luna_segment_path, luna_data=luna_data,
                          savepath=prep_folder,
                          use_existing=use_existing)

    N = len(filelist)
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist