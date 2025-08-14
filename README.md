# cpauger

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/agbleze/cpauger/.github%2Fworkflows%2Fci-cd.yml)
![GitHub Tag](https://img.shields.io/github/v/tag/agbleze/cpauger)
![GitHub Release](https://img.shields.io/github/v/release/agbleze/cpauger)
![GitHub License](https://img.shields.io/github/license/agbleze/cpauger)

# Overview

cpauger package provides functionality for cropping objects in source image(s) and pasting in  destination image(s) (background images) and generating coco annotations for the resulting data.

It provides the following functionality

- Cropping and pasting objects
- Specify location to paste the object at in destination image
- Resizing object to be pasted
- Specify for each object, number to paste into a given image
- Generating random images and annotations
- Visualing the pasted objects 

# Why the need

Training of computer vision models to generalize well requires providing both diverse objects and different backgrounds on which the objects are captured. In reality, this ideal scenario is plagued by the following scenario

- Lack of data on target object in desired scene that model is expected to make prediction. In this case you want to predict objects in a scene that you have not yet captured the object in
- Balance target object inbalance for minority objects in the image

To handle any of the above scenario, copy-paste augmentation has proven worthwhile. Essentially, you crop and paste the object on the background image of interest for model training.


## Installation

```bash
$ pip install cpauger
```

## Usage

To run the complete function for copy-paste augmentation and generate annotation,
you need images in a folder (source) and their annotation in COCO format, a list of images from which objects are to be cropped, objects to crop, images in a folder to use as background images.

Use crop_paste_obj function to run copy-paste augmentation as follows:

```python
from cpauger.generate_coco_ann import generate_random_images_and_annotation
from cpauger.augment_image import crop_paste_obj

source_img_dir = "source_imgs"
destination_img_dir = "dest_imgs

src_imgs, src_ann = generate_random_images_and_annotation(image_height=224, image_width=224,
                                                        number_of_images=100, 
                                                        output_dir=source_img_dir,
                                                        img_ext ="jpg",
                                                        image_name="src_img",
                                                        parallelize=True,
                                                        save_ann_as="src_annotation.json",
                                                        )


dest_imgs, _ = generate_random_images_and_annotation(image_height=224, image_width=224,
                                                    number_of_images=100, 
                                                    output_dir=destination_img_dir,
                                                    img_ext ="jpg",
                                                    image_name="dest_img",
                                                    parallelize=True,
                                                    save_ann_as= "dest_annotation.json",
                                                    )

crop_paste_obj(object_to_cropped=["object_1"], 
            imgnames_for_crop=src_imgs,
            img_dir=source_img_dir,
            coco_ann_file="src_annotation.json", 
            bkgs=dest_imgs, 
            objs_paste_num={"object_1":1},
            output_img_dir="cpaug_output",
            save_coco_ann_as="cpaug_ann.json",
            min_x=None, min_y=None,
            max_x=None, max_y=None,
            sample_location_randomly=True,
            visualize_dir="viz_output_dir
            )
```

For complete tutorials on several other utilities provided by the package, refer to ![Documentation](https://cpauger.readthedocs.io/en/stable/)


## Test code

To run unit test for cpauger core functionalities, use pytest after cloning the repo and run the command.

```pytest```


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cpauger` was created by linus agbleze. It is licensed under the terms of the MIT license.

## Credits

`cpauger` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
