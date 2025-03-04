import os
import glob
import re
import shutil

def reorganize_data(root, save_folder):
    content_img_list = [p for p in glob.glob(root+'/*.jpg') if not re.search('thumb', p)]
    probe_img_list = [p for p in glob.glob(root+'/probes/*.jpg') if re.search('chrome256', p)]
    content_img_list.sort()
    probe_img_list.sort()

    for content_img in content_img_list:
        content_name = re.sub('/dir_', '_', content_img)
        shutil.copyfile(content_img, f'{save_folder}/{content_name}')

    # for probe_img in probe_img_list:
    #     probe_name = re.sub('/probes/dir_', '_', re.sub('everett_', '', probe_img))
    #     shutil.copyfile(probe_img, f'{save_folder}/{probe_name}')

def main(save_folder):
    folder_list = [f for f in glob.glob('*') if os.path.isdir(f)]
    folder_list.sort()

    for folder in folder_list:
        if folder not in ['test_1', save_folder]:
            print(f'processing folder {folder}...')
            reorganize_data(folder, save_folder)


if __name__ == '__main__':
    save_folder = 'train'
    os.makedirs(save_folder, exist_ok=True)
    main(save_folder)

    img_list = glob.glob(f'{save_folder}/*.jpg')
    img_list.sort()
    print(f'total images: {len(img_list)}')   # should be 24625
    # print(img_list)


