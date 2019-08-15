import pydicom
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mask_functions import mask2rle, rle2mask
from inference import deboning_512
from pathlib import Path
import PIL
import numpy as np
import fastai
from fastai.vision import *
from torchvision.utils import save_image

def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()

for file_path in glob.glob('data/siim_data/dicom-images-train/*/*/*.dcm'):
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    plot_pixel_array(dataset)
    break

#
# plot some images with masks
#
df = pd.read_csv('data/siim_data/train-rle.csv', header=None,index_col=0)

start = 60   # Starting index of images
num_img = 8 # Total number of images to show
fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
for q, file_path in enumerate(glob.glob('data/siim_data/dicom-images-train/*/*/*.dcm')[start:start+num_img]):
    dataset = pydicom.dcmread(file_path)
    #print(file_path.split('/')[-1][:-4])
    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)
    if df.loc[file_path.split('/')[-1][:-4],1] != ' -1':
        mask = rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T
        ax[q].set_title('See Marker')
        ax[q].imshow(mask, alpha=0.3, cmap="Reds")
    else:
        ax[q].set_title('Nothing to see')


# process all the images using the inference module
path_train_512 = Path('data/siim_data/train_512')
path_masks_512 = Path('data/siim_data/masks_512')
foundOne = False
for q, file_path in enumerate(glob.glob('data/siim_data/dicom-images-train/*/*/*.dcm')):
    dataset = pydicom.dcmread(file_path)
    img = PIL.Image.fromarray(dataset.pixel_array).resize([512, 512], resample=PIL.Image.BILINEAR)
    dest = Path(path_train_512/Path(file_path).name).with_suffix('.png')
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    dest_mask = Path(path_masks_512/Path(file_path).name).with_suffix('.png')
    dest_mask.parent.mkdir(parents=True, exist_ok=True)
    # create an empty mask and add the ROIs to it one by one (if there are multiple)
    mask = PIL.Image.new('L', img.size)
    try:
        aa = df.loc[file_path.split('/')[-1][:-4],1]
    except KeyError:
        print('KeyError for %s, there is no mask available.' % file_path.split('/')[-1][:-4])
        aa = '-1' # just pretend its an empty mask
    if isinstance(aa, list):
        print("The masks are a list")
        # This does not really happen at all - just an artifact of the wrong naming generated above with .stem (incorrect) now .name (correct).
        foundOne = True
        for row in range(df.loc[file_path.split('/')[-1][:-4]].size):
            aaa = aa[[row]][0]
            print("found a mask number %d" % row)
            if aaa != ' -1' and aaa != '-1':
                mask_tmp = rle2mask(aaa, 1024, 1024).T
                mask_tmp = PIL.Image.fromarray(mask_tmp.astype('uint8'),'L').resize([512,512])
                # add this mask to the big mask
                a = np.array(mask)
                b = np.array(mask_tmp)
                mask = PIL.Image.fromarray(np.maximum(a,b).astype('uint8'), 'L')
    elif isinstance(aa, str) and (aa != ' -1') and (aa != '-1'):
        print("single mask is saved...")
        mask_tmp = rle2mask(aa, 1024, 1024).T
        mask = PIL.Image.fromarray(mask_tmp.astype('uint8'),'L').resize([512,512])
    else:
        print("mask is neither string with some length or list, create an empty mask here")
    print("save mask: %s" % dest_mask)
    mask.save(dest_mask)
    if foundOne:
        break

# Now for each image in train_512 we would like to compute the deboned version.
# (They should all have a mask now.)
# pre = deboning_512('../../../models/deboning_512_120steps')
# img, img_hr = pre.debone('data/input/img_2050.png')
path_train_512 = Path('data/siim_data/train_512')
path_train_deboned_512 = Path('data/siim_data/train_deboned_512')
pre = deboning_512('../../../models/deboning_512_200steps')
il = ImageList.from_folder(path_train_512)

def debone_one_tissue(fn, i, path):
    dest = path/fn.relative_to(path_train_512)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("debone %s" % (fn))
    img, img_db = pre.debone(fn)
    print("save %s" % dest)
    save_image(img_db, dest)
    #out = PIL.Image.fromarray(image2np(img_db.data).astype('uint8'),'L')
    #aa = image2np(img_db.cpu())
    #print(aa.shape)
    #PIL.Image.fromarray(aa.astype('uint8'), 'RGB').save(dest)

# create the deboned version of all the training data
parallel(partial(debone_one_tissue, path=path_train_deboned_512), il.items)

