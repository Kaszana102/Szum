import os
import shutil

src_dir = 'dataset_src/other/animals/animals'
dst_dir = 'dataset_src/other/other'

#for each directory
for directory in os.listdir(src_dir):
    for filename in os.listdir(src_dir+'/'+directory):
        imagesrc = src_dir + '/'+directory + '/' + filename
        imagedst = dst_dir +'/'+ filename
        shutil.copyfile(imagesrc, imagedst)

