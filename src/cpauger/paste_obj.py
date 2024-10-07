from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
from clusteval import clusteval
import pandas as pd
import json
from typing import Union, List, Dict, Tuple
import os
import random


# Function to adjust segmentation based on bbox
def adjust_segmentation(bbox, segmentation):
    x_offset, y_offset = bbox[0], bbox[1]
    adjusted_segmentation = []
    for polygon in segmentation:
        adjusted_polygon = []
        for i in range(0, len(polygon), 2):
            adjusted_polygon.append(polygon[i] + x_offset)
            adjusted_polygon.append(polygon[i + 1] + y_offset)
        adjusted_segmentation.append(adjusted_polygon)
    return adjusted_segmentation


def paste_object(dest_img_path, cropped_objects: Dict[str, List[np.ndarray]], min_x=None, min_y=None, 
                 max_x=None, max_y=None, 
                 resize_w=None, resize_h=None, 
                 sample_location_randomly: bool = True,
                 )->Tuple[np.ndarray, List, List, List]:
    dest_image = cv2.imread(dest_img_path, cv2.IMREAD_UNCHANGED)
    dest_image = cv2.cvtColor(dest_image, cv2.COLOR_BGR2RGB)
    dest_h, dest_w = dest_image.shape[:2]
    if not isinstance(cropped_objects, dict):
        raise ValueError(f"""cropped_objects is expected to be a dictionary of 
                         key being the category_id and value being a list of
                         cropped object image (np.ndarray)
                         """)
    bboxes, segmentations, category_ids = [], [], []
    # Resize the cropped object if resize dimensions are provided
    for cat_id in cropped_objects:
        cat_cropped_objects = cropped_objects[cat_id]
        if not isinstance(cat_cropped_objects, list):
            cat_cropped_objects = [cat_cropped_objects]
        for cropped_object in cat_cropped_objects:
            print(f"cropped_object: {cropped_object.shape}")
            if sample_location_randomly:
                min_x = random.random()
                max_x = random.uniform(min_x, 1)
                min_y = random.random()
                max_y = random.uniform(min_y, 1)
                
                x = int(min_x * dest_w)
                y = int(min_y * dest_h)
                max_x = int(max_x * dest_w)
                max_y = int(max_y * dest_h)
            else:
                x = int(min_x * dest_w)
                y = int(min_y * dest_h)
                max_x = int(max_x * dest_w)
                max_y = int(max_y * dest_h)
                
            if resize_w and resize_h:
                print(f"cropped_object: {cropped_object} \n")
                obj_h, obj_w = cropped_object.shape[:2]
                aspect_ratio = obj_w / obj_h
                if resize_w / resize_h > aspect_ratio:
                    resize_w = int(resize_h * aspect_ratio)
                else:
                    resize_h = int(resize_w / aspect_ratio)
                resized_object = cv2.resize(cropped_object, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            else:
                resized_object = cropped_object

            # Ensure the resized object fits within the specified area
            obj_h, obj_w = resized_object.shape[:2]
            if obj_w > (max_x - x) or obj_h > (max_y - y):
                scale_x = (max_x - x) / obj_w
                if scale_x <= 0:
                    scale_x = 0.1
                scale_y = (max_y - y) / obj_h
                if scale_y <= 0:
                    scale_y = 0.1
                    
                scale = min(scale_x, scale_y)
                new_w = int(obj_w * scale)
                new_h = int(obj_h * scale)
                resized_object = cv2.resize(resized_object, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create a mask for the resized object
            print(f"resized_object: {resized_object.shape} \n")
            if resized_object.shape[2] == 3:
                resized_object = cv2.cvtColor(resized_object, cv2.COLOR_RGB2RGBA)
                print(f"after resized object to RGBA: {resized_object.shape}")
            mask = resized_object[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)
            resized_object = resized_object[:, :, :3]

            # Define the region of interest (ROI) in the destination image
            roi = dest_image[y:y+resized_object.shape[0], x:x+resized_object.shape[1]]

            # Black-out the area of the object in the ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of the object from the object image
            obj_fg = cv2.bitwise_and(resized_object, resized_object, mask=mask)

            # Put the object in the ROI and modify the destination image
            dst = cv2.add(img_bg, obj_fg)
            dest_image[y:y+resized_object.shape[0], x:x+resized_object.shape[1]] = dst

            # Calculate the bounding box
            bbox = [x, y, resized_object.shape[1], resized_object.shape[0]]
            # Calculate the segmentation 
            # changed mask to resized_object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)
            bboxes.append(bbox)
            # translate segmentation here
            adjusted_segmentation = adjust_segmentation(bbox=bbox, segmentation=segmentation)
            segmentations.append(adjusted_segmentation)
            category_ids.append(int(cat_id))
    return dest_image, bboxes, segmentations, category_ids



def paste_crops_on_bkgs(bkgs, all_crops, objs_paste_num: Dict, output_img_dir, save_coco_ann_as,
                        min_x=None, min_y=None, 
                        max_x=None, max_y=None, 
                        resize_width=None, resize_height=None,
                        sample_location_randomly=True
                        ):
    os.makedirs(output_img_dir, exist_ok=True)
    coco_ann = {"categories": [{"id": obj_idx+1, "name": obj} for obj_idx, obj in enumerate(sorted(objs_paste_num))], 
                "images": [], 
                "annotations": []
                }
    ann_ids = []
    for bkg_idx, bkg in enumerate(bkgs):
        sampled_obj = {obj_idx+1: random.sample(all_crops[obj], int(objs_paste_num[obj])) 
                       for obj_idx, obj in enumerate(sorted(objs_paste_num))
                       }
        dest_img, bboxes, segmasks, category_ids = paste_object(dest_img_path=bkg,  ## showed also return the obj_idx as category_id
                                                                cropped_objects=sampled_obj,
                                                                min_x=min_x, min_y=min_y, max_x=max_x,
                                                                max_y=max_y, resize_h=resize_height,
                                                                resize_w=resize_width,
                                                                sample_location_randomly=sample_location_randomly
                                                                )
        file_name = os.path.basename(bkg)
        img_path = os.path.join(output_img_dir, file_name)
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, dest_img)
        assert(len(bboxes) == len(segmasks) == len(category_ids)), f"""bboxes: {len(bboxes)}, segmasks: {len(segmasks)} and category_ids: {len(category_ids)} are not equal length"""
        img_height, img_width = dest_img.shape[0], dest_img.shape[1]
        img_id = bkg_idx+1
        image_info = {"file_name": file_name, "height": img_height, 
                        "width": img_width, "id": img_id
                        }
        coco_ann["images"].append(image_info)
        
        for ann_ins in range(0, len(bboxes)):
            bbox = bboxes[ann_ins]
            segmask = segmasks[ann_ins]
            ann_id = len(ann_ids) + 1
            ann_ids.append(ann_id)
            category_id = category_ids[ann_ins]
            annotation = {"id": ann_id, 
                          "image_id": img_id, 
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": segmask
                        } 
            coco_ann["annotations"].append(annotation)
    with open(save_coco_ann_as, "w") as filepath:
        json.dump(coco_ann, filepath)            
                
