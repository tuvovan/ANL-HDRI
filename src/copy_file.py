import os
import shutil
import glob


def move(orig, dest):
    for d in os.listdir(orig):
        if not os.path.exists(os.path.join(dest, d)):
            os.makedirs(os.path.join(dest, d))
        f = os.path.join(orig, d, 'ref_hdr.hdr')
        shutil.copyfile(f, os.path.join(dest, d, f.split('/')[-1]))

        f = os.path.join(orig, d, 'input_exp.txt')
        shutil.copyfile(f, os.path.join(dest, d, f.split('/')[-1]))

def move_tif(orig, dest):
    for d in os.listdir(orig):
        if not os.path.exists(os.path.join(dest, d)):
            os.makedirs(os.path.join(dest, d))
        for f in glob.glob(os.path.join(orig, d, 'input_*.tif')):
            if not 'aligned' in f:
                shutil.copyfile(f, os.path.join(dest, d, f.split('/')[-1]))


# move_tif('test/', 'dataset_test/')
move('test/', 'dataset_test/')