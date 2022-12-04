import os
import glob
import random
import argparse

CLASS_NAME = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Q2 and Q3 TODO : select "better" images from Q2 folder
def select_imaegs(image_paths, images_num=200):
    """
    :param image_paths: --> ['your_folder/images1.jpg', 'your_folder/images2.jpg', ...]
    :param images_num: choose the number of images
    :return :
        selected_image_paths = ['your_folder/images10.jpg', 'your_folder/images12.jpg', ...]
    """
    # TODO : select images

    return selected_image_paths

# TODO : split train and val images
def split_train_val_path(all_image_paths, train_val_ratio=0.9):
    """
    :param all_image_paths: all image paths for question in the data folder
    :param train_val_ratio: ratio of image paths used to split training and validation
    :return :
        train_image_paths = ['your_folder/images1.jpg', 'your_folder/images2.jpg', ...]
        val_image_paths = ['your_folder/images3.jpg', 'your_folder/images4.jpg', ...]
    """
    # TODO : split train and val
    train_image_paths = all_image_paths[: int(len(all_image_paths) * train_val_ratio)]  # just an example
    val_image_paths = all_image_paths[int(len(all_image_paths) * train_val_ratio):]  # just an example

    return train_image_paths, val_image_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./data/CityCam', help='path of CityCam datasets folder')
    parser.add_argument('--ques', type=str, default='Q1', choices=['Q1', 'Q2', 'Q3'], help='question in data_folder')
    args = parser.parse_args()
    print(args)

    # Get whole and Test image paths
    all_image_paths = glob.glob(os.path.join(args.data_folder, args.ques, '*', '*.jpg'))
    test_image_paths = glob.glob(os.path.join(args.data_folder, 'test', '*' + os.sep + '*.jpg'))

    # for Q2 and Q3 : select images
    if args.ques == 'Q2' or args.ques == 'Q3':
        selected_image_paths = select_imaegs(all_image_paths, images_num=200)
    else:
        selected_image_paths = all_image_paths
    # split Train and Val
    train_image_paths, val_image_paths = split_train_val_path(selected_image_paths)

    # write train/val/test info
    train_path = os.path.join(args.data_folder, 'train.txt')
    val_path = os.path.join(args.data_folder, 'val.txt')
    test_path = os.path.join(args.data_folder, 'test.txt')
    with open(train_path, 'w') as f:
        for image_path in train_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
    with open(val_path, 'w') as f:
        for image_path in val_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
    with open(test_path, 'w') as f:
        for image_path in test_image_paths:
            f.write(os.path.abspath(image_path) + '\n')

    # write training YAML file
    with open('./data/citycam.yaml', 'w') as f:
        f.write("train: " + os.path.abspath(train_path) + "\n")
        f.write("val: " + os.path.abspath(val_path) + "\n")
        f.write("test: " + os.path.abspath(test_path) + "\n")
        # number of classes
        f.write('nc: 80\n')
        # class names
        f.write('names: ' + str(CLASS_NAME))

    # delete cache
    if os.path.exists(os.path.join(args.data_folder, 'train.cache')):
        os.remove(os.path.join(args.data_folder, 'train.cache'))
    if os.path.exists(os.path.join(args.data_folder, 'val.cache')):
        os.remove(os.path.join(args.data_folder, 'val.cache'))
    """
    if os.path.exists(os.path.join(args.data_folder, 'test.cache')):
        os.remove(os.path.join(args.data_folder, 'test.cache'))
        """
