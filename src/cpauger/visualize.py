import cv2
import json
import os
from pathlib import Path
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import random
import COCO

def visualize_bboxes(annotation_file, image_dir, output_dir):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    category_map = {category['id']: category['name'] for category in coco_data['categories']}

    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_map[category_id]
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        output_path = os.path.join(output_dir, f"bbox_{image_info['file_name']}")
        cv2.imwrite(output_path, image)
        print(f"Bounding boxes visualized for {image_info['file_name']} and saved as {output_path}")


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_bbox_and_polygons(annotation_path, img_dir, 
                           visualize_dir="visualize_bbox_and_polygons"
                           ):
    os.makedirs(visualize_dir, exist_ok=True)
    coco = COCO(annotation_path)
    for id, imginfo in coco.imgs.items():
        file_name = imginfo["file_name"]
        imgid = imginfo["id"]
        ann_ids = coco.getAnnIds(imgIds=imgid)
        anns = coco.loadAnns(ids=ann_ids)
        bboxes = [ann["bbox"] for ann in anns]

        polygons = [ann["segmentation"][0] for ann in anns]
        category_ids = [ann["category_id"] for ann in anns]
        category_names = [coco.cats[cat_id]["name"] for cat_id in category_ids]
        
        image_path = os.path.join(img_dir, file_name)
        
        img = Image.open(image_path).convert("RGBA")
        mask_img = Image.new("RGBA", img.size)
        draw = ImageDraw.Draw(mask_img)
        font = ImageFont.load_default()
        # Draw bounding boxes
        for bbox, polygon, category_name in zip(bboxes, polygons, category_names):
            color = random_color()
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            draw.rectangle(bbox, outline=color, width=2)
            draw.polygon(polygon, outline=color, fill=color + (100,))
            text_position = (bbox[0], bbox[1] - 10)
            draw.text(text_position, category_name, fill=color, font=font)
        blended_img = Image.alpha_composite(img, mask_img)
        final_img = blended_img.convert("RGB")
        # Save the output image
        output_path = os.path.join(visualize_dir, file_name)  # Replace "visualize_bbox_and_polygons" with your desired output directory path  # Ensure that the directory exists before saving the image  # Example: output_path = "output/image_with_bbox_and_polygons.png"  # Save the image in PNG format  # Example: img.save(output_path, format='PNG')  # Save the image in JPEG format  # Example: img.save(output_path, format='JPEG')  # Save the image in GIF format  # Example: img.save(output_path, format='GIF')  # Save the image in TIFF format  # Example: img.save(output_path, format='TIFF')  # Save the image in WebP format  # Example: img.save(output_path, format='WEBP')
        final_img.save(output_path, format='PNG') 


#%% Example usage
if __name__ == '__main__':
    annotation_file = "/home/lin/codebase/merge_coco/cor4_merged_annotations.json"
    valid_coco = "/home/lin/codebase/merge_coco/valid_annotations.json"
    image_dir = "/home/lin/codebase/merge_coco/valid"
    output_dir = "/home/lin/codebase/merge_coco/valid_viz_cor4"
    visualize_bboxes(valid_coco, image_dir, output_dir)

# %%
# flea-beetle-2_jpg.rf.5d03f74e1f36cc80c7606c28f6589420
