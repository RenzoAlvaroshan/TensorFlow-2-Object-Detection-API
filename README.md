# TensorFlow-2-Object-Detection-API

This is my repository for learning how to use TensorFlow 2 Object Detection API. Starting from creating workspaces (folders), installing packages and libraries, downloading the pre-trained from TensorFlow 2 Model Zoo, and lastly train and test the model using batch image and also videos.

Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html


## Sample of the result
Taken using Logitech Brio 4K Pro

Image:

![image](https://user-images.githubusercontent.com/55566616/127762079-5c45bfe7-bcef-4849-b711-abb83e4f6b99.jpeg)


Video:

https://user-images.githubusercontent.com/55566616/126886984-cc2fdf0b-5752-4a7e-980e-63783f4612da.mp4

## Prerequisite

### Installation

### Create a new Anaconda virtual environment

Open a new Terminal window
Type the following command:

```python
C:\Users\renzo>conda create -n tensorflow pip python=3.8
```

The above will create a new virtual environment with name tensorflow

### Activate the Anaconda virtual environment

```
C:\Users\renzo>conda activate tensorflow
```

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

```
(tensorflow) C:\Users\renzo>
```

TensorFlow Installation

```
pip install --ignore-installed --upgrade tensorflow==2.5.0
```

TensorFlow Object Detection API Installation

1. Create a new folder under a path of your choice and name it TensorFlow

Downloading the TensorFlow Model Garden
* Create a new folder under a path of your choice and name it TensorFlow. (e.g. C:\Users\renzo\TensorFlow).
* From your Terminal cd into the TensorFlow directory.
* To download the models you can either use Git to clone the TensorFlow Models repository inside the TensorFlow folder, or you can simply download it as a ZIP and extract its contents inside the TensorFlow folder. To keep things consistent, in the latter case you will have to rename the extracted folder models-master to models.
* You should now have a single folder named models under your TensorFlow folder, which contains another 4 folders as such:

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
   └── ...
```

### Protobuf Installation/Compilation
* Head to the protoc releases page
* Download the latest protoc-*-*.zip release (e.g. protoc-3.12.3-win64.zip for 64-bit Windows)
* Extract the contents of the downloaded protoc-*-*.zip in a directory <PATH_TO_PB> of your choice (e.g. C:\Program Files\Google Protobuf)
* Add <PATH_TO_PB>\bin to your Path environment variable (see Environment Setup)
* In a new Terminal 1, cd into TensorFlow/models/research/ directory and run the following command:

```python
protoc object_detection/protos/*.proto --python_out=.
```

COCO API installation

```python
(tensorflow) C:\Users\renzo>pip install cython
(tensorflow) C:\Users\renzo>pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### Install the Object Detection API

```python
# From within TensorFlow/models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

Test your Installation
To test the installation, run the following command from within Tensorflow\models\research:

```python
# From within TensorFlow/models/research/
python object_detection/builders/model_builder_tf2_test.py
```

### Training Custom Object Detector

create a new folder under TensorFlow and call it workspace. It is within the workspace that we will store all our training set-ups. Now let’s go under workspace and create another folder named training_demo. Now our directory structure should be as so:

```
TensorFlow/
├─ addons/ (Optional)
│  └─ labelImg/
├─ models/
│  ├─ community/
│  ├─ official/
│  ├─ orbit/
│  ├─ research/
│  └─ ...
└─ workspace/
   └─ training_demo/
```

The training_demo folder shall be our training folder, which will contain all files related to our model training. It is advisable to create a separate training folder each time we wish to train on a different dataset. The typical structure for training folders is shown below.

```
training_demo/
├─ annotations/
├─ exported-models/
├─ images/
│  ├─ test/
│  └─ train/
├─ models/
├─ pre-trained-models/
└─ README.md
```

Capturing images

1. Prepare your webcamera (e.g. Logitech Brio 4K Pro)
2. Run capture_image.py

python capture_image.py

3. Capture face images from different angles

![renzo_logitech_60_65](https://user-images.githubusercontent.com/55566616/128579829-88d9fe33-fb1d-432d-b43a-4d2ba037e853.jpeg)

![renzo_logitech_35_110](https://user-images.githubusercontent.com/55566616/128579841-8fbe10a3-447f-420b-8bb7-f1c9c706cce1.jpeg)

![renzo_logitech_15_90](https://user-images.githubusercontent.com/55566616/128579877-05c4cd1c-4f9a-4727-a6d9-c1de894bd7d3.jpeg)

I captured from 0° - 180° with the interval 5° horizontally and 0° - 60° with the interval 5° vertically
in total we can get 431 pictures

Save it into your images folder

### Preparing the dataset

Open terminal / anaconda prompt

1. Download labelImg

```
(tensorflow) C:\Users\renzo>pip install labelImg
```

run labelimg using this command:

```
(tensorflow) C:\Users\renzo>labelImg
```

2. Annotate the faces in your image one by one and give the face a name (choose pascal voc format)
(I recommend you using macro for faster process of labeling images)

<img width="1101" alt="Screen Shot 2021-08-07 at 05 26 27" src="https://user-images.githubusercontent.com/55566616/128579899-129a233b-df58-4619-9a43-3a3565620a06.png">

3. Save all you `.jpg` and `.xml` into the same folder

Partition the Dataset

To make things even tidier, let’s create a new folder TensorFlow/scripts/preprocessing, where we shall store scripts that we can use to preprocess our training inputs. Below is out TensorFlow directory tree structure, up to now:

```
TensorFlow/
├─ addons/ (Optional)
│  └─ labelImg/
├─ models/
│  ├─ community/
│  ├─ official/
│  ├─ orbit/
│  ├─ research/
│  └─ ...
├─ scripts/
│  └─ preprocessing/
└─ workspace/
   └─ training_demo/
```

1. Download the partition data set script here
2. Then, cd into TensorFlow/scripts/preprocessing and run:

```python
(tensorflow) C:\Users\renzo>python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1

# For example
# python partition_dataset.py -x -i C:/Users/renzo/Tensorflow/workspace/training_demo/images -r 0.1
```

This will partition our data with the ratio of 90% train data and 10% test data

### Create Label Map

TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.

Below we show an example label map (e.g label_map.pbtxt), assuming that our dataset contains 2 labels (person name), renzo and adriano:

```
item {
    id: 1
    name: ‘renzo’
}

item {
    id: 2
    name: ‘adriano’
}
```

*assume renzo and adriano is two different person

### Create TensorFlow Records

* Click here to download the above script and save it inside `TensorFlow/scripts/preprocessing`.
* Install the pandas package:

```
conda install pandas # Anaconda
                     # or
pip install pandas   # pip
```

Finally, cd into TensorFlow/scripts/preprocessing and run:

```python
# Create train data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

# Create test data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

# For example
# python generate_tfrecord.py -x C:/Users/renzo/Tensorflow/workspace/training_demo/images/train -l C:/Users/renzo/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/renzo/Documents/Tensorflow/workspace/training_demo/annotations/train.record

# python generate_tfrecord.py -x C:/Users/renzo/Tensorflow/workspace/training_demo/images/train -l C:/Users/renzo/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/renzo/Documents/Tensorflow/workspace/training_demo/annotations/test.record
```
