import os, shutil

def file_backkup(log_dir):
    file_name_list = os.listdir(os.getcwd())
    file_path_list = [os.path.join(os.getcwd(),x) for x in file_name_list if '.py' in x]
    for fname, fpath in zip(file_name_list, file_path_list):
        shutil.copy(src=fpath, dst=os.path.join(log_dir,fname))