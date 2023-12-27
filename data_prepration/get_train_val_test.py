import os
from tqdm import tqdm
import sys
sys.path.append("..")
from stereosdf.datasets.utils.file_io import read_text_lines,list2txt


def get_all_avaiable_fname_from_folder_kitti_360(root_path,saved_fname_path=None):
    '''
    Params: 
    
    Root Path: The KITTI-360 Root Path
    Saved_Fname_Path : saved all fnames contains all KITTI-360 images.
    
    Returns:
    
    Saved Files Contains All KITTI-360 Images Names
    
    '''
    
    image_root = os.path.join(root_path,"image_data/data_2d_raw") 
    saved_folder_name = saved_fname_path[:-len(os.path.basename(saved_fname_path))]
    if not os.path.exists(saved_folder_name):
        os.makedirs(saved_folder_name)
    date_decided_categories = os.listdir(image_root) # dates
    saved_files_name_list =[]
    for idx, dates in tqdm(enumerate(date_decided_categories)):
        date_folder_absolute_path = os.path.join(image_root,dates)
        date_folder_absolute_path = os.path.join(date_folder_absolute_path,"image_00/data_rect")        
        for sub_idx, fname in enumerate(os.listdir(date_folder_absolute_path)):
            saved_left_image_name = os.path.join(date_folder_absolute_path,fname)  
            saved_left_image_name = saved_left_image_name[len(root_path)+1:] # consider `/`
            saved_files_name_list.append(saved_left_image_name)
    
    if saved_fname_path is not None:
        with open(saved_fname_path,'w') as f:
            for ind, fname in enumerate(saved_files_name_list):
                if ind!=len(saved_files_name_list)-1:
                    f.writelines(fname+"\n")
                else:
                    f.writelines(fname)


                
def split_the_train_val_and_test_fnames(fname,uid_val=['03','07'],uid_test=['10'],
                                        saved_val_name=None,saved_test_name=None,
                                        saved_train_name=None):
    
    uid_to_folder_name = {
         '00':"2013_05_28_drive_0000_sync",
         "02":"2013_05_28_drive_0002_sync",
         "03":"2013_05_28_drive_0003_sync",
         "04":"2013_05_28_drive_0004_sync",
         "05": "2013_05_28_drive_0005_sync",
         "06": "2013_05_28_drive_0006_sync",
         "07": "2013_05_28_drive_0007_sync",
         "09": "2013_05_28_drive_0009_sync",
         "10":"2013_05_28_drive_0010_sync"
    }
    
    val_folder_names= [uid_to_folder_name[uid] for uid in uid_val]
    test_folder_names =[uid_to_folder_name[uid] for uid in uid_test]
    
    fnames_all = read_text_lines(fname)
    
    val_fnames_list = []
    test_fnames_list = []
    train_fnames_list = []
    
    
    for idx, line in enumerate(fnames_all):
        categories ='train'
        # check validation
        for val_uids in val_folder_names:
            if val_uids in line:
                val_fnames_list.append(line)
                categories='val'
        for test_uids in test_folder_names:
            if test_uids in line:
                test_fnames_list.append(line)
                categories="test"
        
        if categories=='train':
            train_fnames_list.append(line)
    
    
    list2txt(train_fnames_list,saved_train_name)
    list2txt(val_fnames_list,saved_val_name)
    list2txt(test_fnames_list,saved_test_name)
    

if __name__=="__main__":
    
    # Root Path
    image_path_folder = "/data1/liu/KITTI360/KITTI360"
    # All Files
    saved_kitti_all_fname = '../filenames/kitti_360_all.txt'
    # Train Files
    saved_kitti_train = '../filenames/kitti_360_train.txt'
    # Validation Files
    saved_kitti_validation = '../filenames/kitti_360_val.txt'
    # Test Files
    saved_kitti_test = '../filenames/kitti_360_test.txt'
    
    # generated all the avaialbe fnames.
    get_all_avaiable_fname_from_folder_kitti_360(root_path=image_path_folder,
                                                 saved_fname_path=saved_kitti_all_fname)
    
    
    # Train/Val/Test Split
    split_the_train_val_and_test_fnames(fname=saved_kitti_all_fname,
                                        uid_val=['03','07'],
                                        uid_test=["10"],
                                        saved_test_name=saved_kitti_test,
                                        saved_train_name=saved_kitti_train,
                                        saved_val_name=saved_kitti_validation)
    