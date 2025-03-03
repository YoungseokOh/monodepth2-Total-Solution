import os
from tqdm import tqdm
from os import listdir
from options import MonodepthOptions

class A5_dataset_tools:
    def __init__(self, options):
        self.file_dir = os.path.dirname(__file__)  # the directory that A5_dataset_splits.py resides in
        self.opt = options
        self.train_val_folder = sorted(listdir(self.opt.data_path))

        
    def copy_origin_to_denoise(self):
        A5T_path = os.path.join(self.opt.data_path, self.train_val_folder[0])
        folder_list = os.listdir(os.path.join(self.opt.data_path, self.train_val_folder[0]))
        for i in folder_list:
            temp_path = os.path.join(A5T_path, i)
            temp_list = os.listdir(temp_path)
            if len(temp_list) != 100:
                print(temp_path)
        return 0


    def A5_dataset_split(self):
        dataset_version = self.opt.data_path.split(os.path.sep)[-1]
        concat_train_path = os.path.join(self.opt.data_path, self.train_val_folder[0]) # Train
        concat_val_path = os.path.join(self.opt.data_path, self.train_val_folder[2]) # Validation
        train_folder_list = sorted(listdir(concat_train_path))
        val_folder_list = sorted(listdir(concat_val_path))
        # Creating splits folder if it doesn't exists
        db_version_split_path = os.path.join(os.path.join(self.file_dir, 'splits'), dataset_version)
        os.makedirs(db_version_split_path, exist_ok=True)
        train_txt_name, val_txt_name = 'train_files.txt', 'val_files.txt'
        
        # Making train files
        f_t = open(os.path.join(db_version_split_path, train_txt_name), 'w')
        for folder in tqdm(train_folder_list):
            images = sorted([i for i in listdir(os.path.join(concat_train_path, folder)) if i.endswith('.jpg')])
            for i in range(1, len(images)-2):
                train_file = '{} {}\n'.format(os.path.join(self.train_val_folder[0], folder), os.path.splitext(images[i])[0])
                f_t.write(train_file)
        f_t.close()

        # Making val files
        f_v = open(os.path.join(db_version_split_path, val_txt_name), 'w')
        for folder in tqdm(val_folder_list):
            images = sorted([i for i in listdir(os.path.join(concat_val_path, folder)) if i.endswith('.jpg')])
            for i in range(1, len(images)-2):
                val_file = '{} {}\n'.format(os.path.join(self.train_val_folder[2], folder), os.path.splitext(images[i])[0])
                f_v.write(val_file)
        f_t.close()
        print('Work is done.')


if __name__ == "__main__":
    options = MonodepthOptions()
    tool = A5_dataset_tools(options.parse())
    tool.A5_dataset_split()
    # copy_origin_to_denoise(options.parse())