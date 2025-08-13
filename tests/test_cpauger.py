from cpauger.paste_obj import paste_object, paste_crops_on_bkgs
from cpauger.generate_coco_ann import (generate_random_images,
                                       generate_random_images_and_annotation
                                        )
from cpauger.crop_obj import collate_all_crops, crop_obj_per_image
from cpauger.augment_image import crop_paste_obj
import pytest
import tempfile
import json
import os
import shutil


@pytest.fixture
def generate_bkg_imgs():
    random_bkg_images = generate_random_images(image_height=124, image_width=124,
                                                number_of_images=10, 
                                                output_dir="random_bkg_images",
                                                #img_ext=None,
                                                image_name="rand_bkg",
                                                parallelize=True
                                                )
    return random_bkg_images

@pytest.fixture
def generate_crop_imgs_and_annotation():
    tempdir = tempfile.TemporaryDirectory()
    img_paths, gen_coco_path = generate_random_images_and_annotation(image_height=124, image_width=124,
                                                                    number_of_images=10, 
                                                                    output_dir=tempdir.name,
                                                                    #img_ext=None,
                                                                    #image_name=None,
                                                                    parallelize=True
                                                                    )
    return img_paths, gen_coco_path, tempdir

@pytest.fixture
def get_all_crops(generate_crop_imgs_and_annotation):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    all_crop_objects = collate_all_crops(object_to_cropped=["object_1", "object_2"], 
                                         imgnames_for_crop=img_paths,
                                        img_dir=tempdir.name, 
                                        coco_ann_file=coco_path
                                        )
    if os.path.exists(coco_path):
        os.remove(coco_path)
    return all_crop_objects

def test_generate_random_images():
    tempdir = tempfile.TemporaryDirectory()
    output_dir = tempdir.name
    random_bkg_images = generate_random_images(image_height=124, image_width=124,
                                                number_of_images=10, 
                                                output_dir=output_dir,
                                                #img_ext=None,
                                                image_name="rand_bkg",
                                                parallelize=True
                                                )
    assert len(random_bkg_images) == 10, "Wrong number of random images generated"
    tempdir.cleanup()

def test_generate_random_images_and_annotation():
    tempdir = tempfile.TemporaryDirectory()
    output_dir = tempdir.name
    res= generate_random_images_and_annotation(image_height=124, 
                                                image_width=124,
                                                number_of_images=10, 
                                                output_dir=output_dir,
                                                #img_ext=None,
                                                #image_name=None,
                                                parallelize=True
                                                )
    assert len(res) == 2, f"{len(res)} objects were return instead of 2 for image list and annotation"
    assert res[0] is not None, "FAILD! No images were generated"
    assert res[1] is not None, "FAILD! No annotation file generated"
    tempdir.cleanup()
   
def test_paste_object(get_all_crops, generate_bkg_imgs):
    objects = get_all_crops
    dest_img_path = generate_bkg_imgs[0]
    res = paste_object(dest_img_path, {"1": objects["object_1"]})
    assert len(res) == 4, f"{len(res)} were returned instead of 4"
    for _ in res:
        assert _ is not None, f"paste_object returned an empty object for at least one of its returned object"
    if os.path.exists(os.path.dirname(dest_img_path)):
        shutil.rmtree(os.path.dirname(dest_img_path))


def test_paste_crops_on_bkgs(get_all_crops, generate_bkg_imgs):
    bkgs = generate_bkg_imgs
    all_crop_objects = get_all_crops
    tempdir = tempfile.TemporaryDirectory()
    #save_coco_ann_tempfile = tempfile.TemporaryFile(suffix=".json")
    #save_coco_ann_as = save_coco_ann_tempfile.name
    #print(f"save_coco_ann_as:  {save_coco_ann_as}")
    save_coco_ann_as = "save_coco_ann_as.json"
    paste_crops_on_bkgs(bkgs=bkgs, all_crops=all_crop_objects, 
                        objs_paste_num={"object_1": 1},
                        output_img_dir=tempdir.name,
                        save_coco_ann_as=save_coco_ann_as,
                        sample_location_randomly=True,
                        resize_height=50, 
                        resize_width=50
                        )
    with open(save_coco_ann_as, "r") as filepath:
        cocodata = json.load(filepath)
    check_keys = ["images", "annotations", "categories"]
    for annkey in check_keys:
        assert annkey in cocodata, f"FAILED: {annkey} result annotation after running paste_crops_on_bkgs does not contain {annkey} field."
    #save_coco_ann_tempfile.cleanup()    
    if os.path.exists(save_coco_ann_as):
        os.remove(save_coco_ann_as)
        
    shutil.rmtree(os.path.dirname(bkgs[0]))

def test_crop_obj_per_image(generate_crop_imgs_and_annotation):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    cropped_obj = crop_obj_per_image(obj_names=["object_1"], 
                                    imgname=img_paths[0], 
                                    img_dir=tempdir.name,
                                    coco_ann_file=coco_path
                                    )
    assert cropped_obj is not None, f"FAILED crop_obj_per_image: No object cropped"
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)

def test_collate_all_crops(generate_crop_imgs_and_annotation):
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    all_crop_objects = collate_all_crops(object_to_cropped=["object_1", "object_2"], 
                                         imgnames_for_crop=img_paths,
                                        img_dir=tempdir.name, 
                                        coco_ann_file=coco_path
                                        )
    assert all_crop_objects is not None, "FAILED: No objects were cropped and collated"
    tempdir.cleanup()
    if os.path.exists(coco_path):
        os.remove(coco_path)


def test_crop_paste_obj(generate_bkg_imgs, generate_crop_imgs_and_annotation):
    bkgs = generate_bkg_imgs
    img_paths, coco_path, tempdir = generate_crop_imgs_and_annotation
    bkgs_paste_tempdir = tempfile.TemporaryDirectory()
    #save_tempfile = tempfile.TemporaryFile(suffix=".json")
    save_coco_ann_as = "save_coco_ann_as.json"
    vis_tempdir = tempfile.TemporaryDirectory()
    crop_paste_obj(object_to_cropped=["object_1"], 
                    imgnames_for_crop=img_paths,
                    img_dir=tempdir.name, #"random_images",
                    coco_ann_file=coco_path, 
                    bkgs=bkgs, 
                    objs_paste_num={"object_1":1},
                    output_img_dir=bkgs_paste_tempdir.name, #"random_cpaug", 
                    save_coco_ann_as=save_coco_ann_as, #"rand_cpaug_ann.json",
                    min_x=None, min_y=None, 
                    max_x=None, max_y=None, 
                    #resize_width=50, resize_height=50,
                    sample_location_randomly=True,
                    visualize_dir=vis_tempdir.name #"rand_cpaug_visualize_bbox_and_polygons"
                    )
    tempdir.cleanup()
    bkgs_paste_tempdir.cleanup()
    vis_tempdir.cleanup()
    files_to_rm = [coco_path, save_coco_ann_as]
    for file in files_to_rm:
        if os.path.exists(file):
            os.remove(file)
    shutil.rmtree(os.path.dirname(bkgs[0]))
    
