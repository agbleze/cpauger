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


def crop_obj_per_image(obj_names: list, imgname: Union[str, List], img_dir,
                       coco_ann_file: str
                       ) -> Union[Dict[str,List], None]:
    cropped_objs_collection = {}
    # get objs in image
    with open(coco_ann_file, "r") as filepath:
        coco_data = json.load(filepath)
        
    categories = coco_data["categories"]
    category_id_to_name_map = {cat["id"]: cat["name"] for cat in categories}
    category_name_to_id_map = {cat["name"]: cat["id"] for cat in categories}
    
    coco = COCO(coco_ann_file)
    images = coco_data["images"]
    image_info = [img_info for img_info in images if img_info["file_name"]==imgname][0]
    image_id = image_info["id"]
    image_height = image_info["height"]
    image_width = image_info["width"]
    annotations = coco_data["annotations"]
    img_ann = [ann_info for ann_info in annotations if ann_info["image_id"]==image_id]
    img_catids = set(ann_info["category_id"] for ann_info in img_ann)
    img_objnames = [category_id_to_name_map[catid] for catid in img_catids]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)
    objs_to_crop = set(img_objnames).intersection(set(obj_names))
    if objs_to_crop:
        for objname in obj_names:
            print(f"objname: {objname} \n")
            object_masks = []
            if objname in img_objnames:
                obj_id = category_name_to_id_map[objname]
                for ann in img_ann:
                    if ann["category_id"] == obj_id:
                        mask = coco.annToMask(ann)                        
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cropped_object = image[y:y+h, x:x+w]
                            mask_cropped = mask[y:y+h, x:x+w]
                            cropped_object = cv2.bitwise_and(cropped_object, cropped_object, 
                                                             mask=mask_cropped)
                            # Remove the background (set to transparent)
                            cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGBA)
                            print(f"mask_cropped: {mask_cropped.shape} \n")
                            print(f"new mask_cropped[:,:] {mask_cropped[:,:].shape} \n")
                            cropped_object[:, :, 3] = mask_cropped * 255
                            object_masks.append(cropped_object)
                if objname not in cropped_objs_collection.keys():
                    cropped_objs_collection[objname] = object_masks
                else:
                    for each_mask in object_masks:
                        cropped_objs_collection[objname].append(each_mask)
            
            
    return cropped_objs_collection


def collate_all_crops(object_to_cropped, imgnames_for_crop, img_dir,
                      coco_ann_file
                      ):
    all_crops = {}
    for img in imgnames_for_crop:
        crop_obj = crop_obj_per_image(obj_names=object_to_cropped, 
                                      imgname=img, 
                                    img_dir=img_dir,
                                    coco_ann_file=coco_ann_file
                                    )
        for each_object in crop_obj.keys():
            if each_object not in all_crops.keys():
                all_crops[each_object] = crop_obj[each_object]
            else:
                cpobjs = crop_obj[each_object]
                if all_crops[each_object] is None:
                    all_crops[each_object] = cpobjs
                else:
                    for idx, cpobj in enumerate(cpobjs): 
                        all_crops[each_object].append(cpobj)           
                    
    return all_crops