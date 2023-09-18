import pandas as pd
import numpy as np
from ensemble_boxes import *
from glob import glob
import copy
from tqdm import tqdm
import shutil
import os
import json

import argparse


def convert_json2df(INPUT_DIR, IMAGE_INFO_PATH, TARGET_LIST):
    height_dict = pd.read_csv(f"{IMAGE_INFO_PATH}/{TARGET_LIST}.csv").to_dict('records')
    fnl_dict={}
    for ix,i in enumerate(height_dict):
        fnl_dict[i['image_id']] = [i['width'],i['height'],i['width'],i['height']]
    
    with open(f"{INPUT_DIR}/{TARGET_LIST}.json", "r") as json_db:
        df = json.load(json_db)
    df = pd.DataFrame(df) # 원본은 coco format이므로 x, y, w, h임. 따라서 xmin, ymin, xmx, ymax로 convert
    bbox_df = pd.DataFrame(df['bbox'].to_list(), columns=['xmin','ymin', 'xmax', 'ymax'])
    bbox_df['xmax'] = bbox_df['xmin'] + bbox_df['xmax']
    bbox_df['ymax'] = bbox_df['ymin'] + bbox_df['ymax']
    df = pd.concat([df, bbox_df], axis = 1)
    df.drop('bbox', axis = 1, inplace= True)
    df['image_id'] = df['image_id'].astype('int')
    
    image_info = pd.read_csv(f"{IMAGE_INFO_PATH}/{TARGET_LIST}.csv")
    df = df.merge(image_info, on = 'image_id', how = 'left')
    
    return df, fnl_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir1',
        help='Output json file to ensemble'
    )
    parser.add_argument(
        '--input_dir2', 
        help='Output json file to ensemble'
    )
    parser.add_argument(
        '--image_info_path', 
        help='Information of image width, height'
    )
    parser.add_argument(
        '--output_dir', 
        help='output directory to save weighted box fusion')

    args = vars(parser.parse_args())

    INPUT_DIR1 = args['input_dir1']
    INPUT_DIR2 = args['input_dir2']
    IMAGE_INFO_PATH = args['image_info_path']
    OUT_DIR = args['output_dir']


target_lists = ["iid_test", "occlusion", "pose", "shape", "texture", "weather", "context"]
for target_list in target_lists:
    print(target_list)
    df1, fnl_dict = convert_json2df(INPUT_DIR1,IMAGE_INFO_PATH, target_list)
    df2, _ = convert_json2df(INPUT_DIR2,IMAGE_INFO_PATH, target_list)
    subs = [df1, df2]

    boxes_dict = {}
    scores_dict = {}
    labels_dict = {}
    whwh_dict = {}

    for i in tqdm(subs[0].image_id.unique()):
        if not i in boxes_dict.keys():
            boxes_dict[i] = []
            scores_dict[i] = []
            labels_dict[i] = []
            whwh_dict[i] = []

        size_ratio = fnl_dict.get(i)
        whwh_dict[i].append(size_ratio) 
        tmp_df = [subs[x][subs[x]['image_id']==i] for x in range(len(subs))]
        
        for x in range(len(tmp_df)):
            boxes_dict[i].append(((tmp_df[x][['xmin','ymin','xmax','ymax']].values)/size_ratio).tolist())
            scores_dict[i].append(tmp_df[x]['score'].values.tolist())
            labels_dict[i].append(tmp_df[x]['category_id'].values.tolist())
            
    weights = [1.5, 1]
    iou_thr = 0.25
    skip_box_thr = 0.0
    sigma = 0.1

    fnl = {}

    for i in tqdm(boxes_dict.keys()):
        
        
        boxes, scores, labels = weighted_boxes_fusion(boxes_dict[i], scores_dict[i], labels_dict[i],\
                                                    weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr) 
        
        if not i in fnl.keys():
            fnl[i] = {'boxes':[],'scores':[],'labels':[]}
            
        fnl[i]['boxes'] = boxes*whwh_dict[i]
        fnl[i]['scores'] = scores
        fnl[i]['labels'] = labels
        
    pd_form = []
    for i in fnl.keys():
        b = fnl[i]
        for j in range(len(b['boxes'])):
            pd_form.append([i,int(b['labels'][j]),round(b['scores'][j],2),\
                            int(b['boxes'][j][0]),int(b['boxes'][j][1]),\
                            int(b['boxes'][j][2]),int(b['boxes'][j][3])])
            
    final_df = pd.DataFrame(pd_form,columns = ['image_id','category_id','score','xmin','ymin','xmax','ymax'])
    final_df = final_df.drop_duplicates(keep = 'first')

    # xmin, ymin, xmax, ymax --> x, y, w, h
    final_df['xmax'] = final_df['xmax'] - final_df['xmin']
    final_df['ymax'] = final_df['ymax'] - final_df['ymin'] 

    final_df['bbox'] = final_df[['xmin','ymin','xmax','ymax']].values.tolist()
    final_df.drop(['xmin','ymin','xmax','ymax'],axis = 1, inplace= True)
    final_df['image_id'] = final_df['image_id'].astype('str')
    
    
    final_df = final_df[['bbox', 'image_id', 'score', 'category_id']]

    result = final_df.to_json(orient="records")
    parsed = json.loads(result)

    with open(f"{OUT_DIR}/{target_list}.json", 'w', encoding='utf-8') as file:
        json.dump(parsed, file)