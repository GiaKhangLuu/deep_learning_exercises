import os
import scipy.io
import csv
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Function to convert a 2D numpy array to a tuple of 4 values
def array_to_tuple(arr):
    return tuple(arr[0])

# Function to process each folder
def process_folder(folder_path, output_dir):
    folder_name = os.path.basename(folder_path)
    classes = ['airplane', 'face', 'motorcycle', 'accordion', 'ant', 'buddha', 'camera', 'octopus']
    target_class = None

    if folder_name == 'Airplanes_Side_2':
        target_class = classes[0]
    elif folder_name == 'Faces_2':
        target_class = classes[1]
    elif folder_name == 'Motorbikes_16':
        target_class = classes[2]
    elif folder_name == 'accordion':
        target_class = classes[3]
    elif folder_name == 'ant':
        target_class = classes[4]
    elif folder_name == 'buddha':
        target_class = classes[5]
    elif folder_name == 'camera':
        target_class = classes[6]
    elif folder_name == 'octopus':
        target_class = classes[7]
    else:
        raise ValueError('Invalid folder name')

    csv_file_path = output_dir / (target_class + '.csv')
    files = glob(str(folder_path / '*.mat'))

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Iterate through .mat files in the folder
        for file in tqdm(files):
            file_name = os.path.basename(file).replace('annotation', 'image').replace('mat', 'jpg')
            if file.endswith('.mat'):
                mat_data = scipy.io.loadmat(file)
                
                if 'box_coord' in mat_data:
                    box_coord = mat_data['box_coord']
                    values = array_to_tuple(box_coord)
                    # Default format is (y1, y2, x1, x2)
                    # Change to (x1, y1, x2, y2)
                    values = [values[2], values[0], values[3], values[1]]
                    writer.writerow([file_name] + list(values) + [target_class])

# Main function to iterate through all folders
def main(input_dir, output_dir):
    objects_used = ['Airplanes_Side_2', 'Faces_2', 'Motorbikes_16', 'accordion', 'ant', 'buddha', 'camera', 'octopus']
    for object in tqdm(objects_used):
        older_path = os.path.join(input_dir, object)
        folder_path = input_dir / object
        process_folder(folder_path, output_dir)

if __name__ == '__main__':
    # Set your input and output directories
    input_directory = Path('./dataset/annotations')
    output_directory = Path('./dataset/annotations')
    main(input_directory, output_directory)
