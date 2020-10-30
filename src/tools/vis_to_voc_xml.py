import cv2
import os
from tqdm import tqdm
import numpy as np

# root = '/Users/olega/Downloads/VisDrone2019-DET-train'
root = '/home/prototype/Downloads/CenterNet-master/data/VisDrone2020-DET/VisDrone2019-DET-train'
input_img_folder = os.path.join(root, 'images')
input_ann_folder = os.path.join(root, 'annotations')
output_ann_folder = os.path.join(root, 'annotations_new')
output_img_folder = os.path.join(root, 'images_new')

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_ann_folder, exist_ok=True)

image_list = os.listdir(input_img_folder)
annotation_list = os.listdir(input_ann_folder)

label_dict = {
    "0": "Ignore",
    "1": "Pedestrian",
    "2": "People",
    "3": "Bicycle",
    "4": "Car",
    "5": "Van",
    "6": "Truck",
    "7": "Tricycle",
    "8": "Awning-tricycle",
    "9": "Bus",
    "10": "Motor",
    "11": "Others"
}

thickness = 2
color = (255, 0, 0)
count = 0


def object_string(label, bbox):
    req_str = '''
	<object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
	'''.format(label, bbox[0], bbox[1], bbox[2], bbox[3])
    return req_str


for annotation in tqdm(annotation_list):
    annotation_path = os.path.join(os.getcwd(), input_ann_folder, annotation)
    xml_annotation = annotation.split('.txt')[0] + '.xml'
    xml_path = os.path.join(os.getcwd(), output_ann_folder, xml_annotation)
    img_file = annotation.split('.txt')[0] + '.jpg'
    img_path = os.path.join(os.getcwd(), input_img_folder, img_file)
    output_img_path = os.path.join(os.getcwd(), output_img_folder, img_file)
    img = cv2.imread(img_path)
    try:
        annotation_string_init = '''
        <annotation>
            <folder>annotations</folder>
            <filename>{}</filename>
            <path>{}</path>
            <source>
                <database>Unknown</database>
            </source>
            <size>
                <width>{}</width>
                <height>{}</height>
                <depth>{}</depth>
            </size>
        <segmented>0</segmented>'''.format(img_file, img_path, img.shape[0], img.shape[1], img.shape[2])
    except Exception as e:
        print(img_path)
        continue

    file = open(annotation_path, 'r')
    lines = file.readlines()
    for line in lines:
        new_line = line.strip('\n').split(',')
        new_coords_min = (int(new_line[0]), int(new_line[1]))
        new_coords_max = (int(new_line[0]) + int(new_line[2]), int(new_line[1]) + int(new_line[3]))
        bbox = (
        int(new_line[0]), int(new_line[1]), int(new_line[0]) + int(new_line[2]), int(new_line[1]) + int(new_line[3]))
        if new_line[5] != '0' and new_line[5] != '11':
            label = label_dict.get(new_line[5])
        else:
            continue
        req_str = object_string(label, bbox)
        annotation_string_init = annotation_string_init + req_str
    # cv2.rectangle(img, new_coords_min, new_coords_max, color, thickness)
    cv2.imwrite(output_img_path, img)
    annotation_string_final = annotation_string_init + '</annotation>'
    f = open(xml_path, 'w')
    f.write(annotation_string_final)
    f.close()
    count += 1
print('[INFO] Completed {} image(s) and annotation(s) pair'.format(count))
