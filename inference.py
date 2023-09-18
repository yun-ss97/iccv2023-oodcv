import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
import json
import os

from models.create_fasterrcnn_model import create_model
from utils_.annotations import (
    inference_annotations, convert_detections
)
from utils_.general import set_infer_dir
from utils_.transforms import infer_transforms, resize


def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        default='data/phase2-test-images',
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '--data', 
        default='data_configs/oodcv.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_vitdet',
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', 
        default='',
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', 
        default=0.5, 
        type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show',  
        action='store_true',
        default=False,
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', 
        dest='mpl_show',
        default=False, 
        action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=256,
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--square-img',
        dest='square_img',
        action='store_true',
        help='whether to use square image resize, else use aspect ratio resize'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    parser.add_argument(
        '--track',
        action='store_true'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['data'] is not None:
        with open(args['data']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    dir_name = args['weights'].split('/')[2]
    OUT_DIR = set_infer_dir(dir_name)

    # Load the pretrained model
    if args['weights'] is None:
        # If the config file is still None, 
        # then load the default one for COCO.
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
            model, coco_model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['data']['NC']
            CLASSES = checkpoint['data']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.transform.min_size = (args['imgsz'], )

    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) 
  

    output_type = ['iid_test', 'context', 'occlusion', 'pose', 'shape', 'texture', 'weather']
    for idx, output_name in enumerate(output_type):
        os.makedirs(f"{OUT_DIR}/{output_name}", exist_ok=True)
    
        DIR_TEST = os.path.join(args['input'],output_name)

        test_images = collect_all_images(DIR_TEST)
        print(f"Test  instances: {len(test_images)}")

        # Define the detection threshold any detection having
        # score below this will be discarded.
        detection_threshold = args['threshold']
        
        # To count the total number of frames iterated through.
        frame_count = 0
        # To keep adding the frames' FPS.
        total_fps = 0

        total_json = []

        for i in range(len(test_images)):
            # Get the image file name for saving output later on.
            image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]

            try:
                orig_image = cv2.imread(test_images[i])
                frame_height, frame_width, _ = orig_image.shape
            except:
                print(image_name)
                continue

            if args['imgsz'] != None:
                RESIZE_TO = args['imgsz']
            else:
                RESIZE_TO = frame_width
            # orig_image = image.copy()
            image_resized = resize(
                orig_image, RESIZE_TO, square=args['square_img']
            )
            image = image_resized.copy()
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image.to(DEVICE))
            end_time = time.time()
                
            # Get the current fps.
            fps = 1 / (end_time - start_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            # Load all detection to CPU for further operations.
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                draw_boxes, pred_classes, scores, labels = convert_detections(
                    outputs, detection_threshold, CLASSES, args
                )
                orig_image, json_file = inference_annotations(
                    image_name,
                    draw_boxes, 
                    pred_classes, 
                    scores,
                    CLASSES,
                    COLORS, 
                    orig_image, 
                    image_resized,
                    labels,
                    output_name,
                    args
                )
                total_json.extend(json_file)
                if args['show']:
                    cv2.imshow('Prediction', orig_image)
                    cv2.waitKey(1)
                if args['mpl_show']:
                    plt.imshow(orig_image[:, :, ::-1])
                    plt.axis('off')
                    plt.show()

            # if i+1 < 50:
                # cv2.imwrite(f"{OUT_DIR}/{output_name}/{image_name}.jpg", orig_image)
                # print(f"Image {i+1} done...")
                # print('-'*50)
        with open(os.path.join(OUT_DIR, f'{output_name}.json'), 'w') as f:
                json.dump(total_json, f)
        print(f'Saved {output_name}.json!')

        print('TEST PREDICTIONS COMPLETE')
        cv2.destroyAllWindows()
        # Calculate and print the average FPS.
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        print('-'*50)

if __name__ == '__main__':
    args = parse_opt()
    main(args)