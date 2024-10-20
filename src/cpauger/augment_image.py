from .paste_obj import paste_crops_on_bkgs
from .visualize import draw_bbox_and_polygons
from .crop_obj import collate_all_crops

def crop_paste_obj(object_to_cropped, imgnames_for_crop, img_dir,
                    coco_ann_file, bkgs, objs_paste_num,
                    output_img_dir, save_coco_ann_as,
                    min_x=None, min_y=None, 
                    max_x=None, max_y=None, 
                    resize_width=None, resize_height=None,
                    sample_location_randomly=True,
                    visualize_dir="cpaug_visualize_bbox_and_polygons"
                    ):
    all_crop_objects = collate_all_crops(object_to_cropped=object_to_cropped, 
                                         imgnames_for_crop=imgnames_for_crop,
                                        img_dir=img_dir, coco_ann_file=coco_ann_file
                                        )
    paste_crops_on_bkgs(bkgs=bkgs, all_crops=all_crop_objects, 
                        objs_paste_num=objs_paste_num,
                        output_img_dir=output_img_dir,
                        save_coco_ann_as=save_coco_ann_as,
                        sample_location_randomly=sample_location_randomly,
                        min_x=min_x, min_y=min_y, max_x=max_x, 
                        max_y=max_y, resize_height=resize_height, 
                        resize_width=resize_width, 
                        )
    draw_bbox_and_polygons(annotation_path=save_coco_ann_as, 
                            img_dir=output_img_dir, 
                            visualize_dir=visualize_dir
                            )

