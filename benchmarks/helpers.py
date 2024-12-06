import os, shutil
import glob

def get_data_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, "data")
    return data_dir


def clear_previous_results(dir: str):
    output_dir = os.path.join(get_data_dir(), dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_image_paths(dir: str):
    input_dir = os.path.join(get_data_dir(), dir)
    image_paths = glob.glob(f"{input_dir}/*")
    return image_paths
