import os
import shutil

def file_check(original_dir, new_dir):
    
    if os.path.isdir(original_dir):
        for file in os.listdir(original_dir):
            shutil.move(os.path.join(original_dir, file), os.path.join(new_dir, file))
          
        if len(os.listdir(original_dir)) == 0:
            shutil.rmtree(original_dir) 
            
def remove_unneeded_yamls(demucs_dir):
    
    for file in os.listdir(demucs_dir):
        if file.endswith('.yaml'):
            if os.path.isfile(os.path.join(demucs_dir, file)):
                os.remove(os.path.join(demucs_dir, file))
    
def remove_temps(remove_dir):
    
    if os.path.isdir(remove_dir):
        try:
            shutil.rmtree(remove_dir) 
        except Exception as e:
            print(e)
            