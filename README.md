# Fruit-Super-Resolute-Classifer

## Instructions
There are 3 components to this project that need to be executed in this order:

1. Cropping
2. Upsample using SRGAN
3. Homography transformation

Here is how to run each step:

Install all required files using `requirements.txt`
### Cropping:
```shell
cd src
python fruit_box.py --input=PATH_TO_IMAGE
```
Choose the bounding box, press C and note the bounding box coordinates.

### SRGAN
```shell
cd src/leftthomas_SRGAN
python test_image.py --image_name=PATH_TO_IMAGE --test_mode=GPU/CPU --save_path=PATH_TO_SAVE_FILE
```
Generator will create a file in `save_path`.

### Homography
```shell
cd src
python transformationFinder.py --left_image=PATH_TO_IMAGE_WITH_FRUIT --right_image=PATH_TO_IMAGE_WITHOUT_FRUIT --bounding_box=BOUNDING_BOX(FORMAT: y1, y2, x1, x2) --upscaled_fruit=PATH_TO_UPSAMPLED_FRUIT
```
This adds the upsampled image into the image without the fruit with correct orientation, and shows it using pyplot.

### SRGAN
Trained https://github.com/leftthomas/SRGAN on dataset https://data.vision.ee.ethz.ch/cvl/DIV2K/

### Data Sets:
https://github.com/Horea94/Fruit-Images-Dataset : Hand Picked 10 fruits for classification purposes

### Tutorials and Documentation Followed:
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
