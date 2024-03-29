# Guide for Creating Face Recognition using TensorFlow 2 Object Detection API

This is my repository for learning how to create custom Face Recognition using TensorFlow 2 Object Detection API. Starting from creating workspaces (folders), installing packages and libraries, downloading the pre-trained from TensorFlow 2 Model Zoo, and finally, train and test the model using batch images and videos.

Reference: 

Lyudmil Vladimirov (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html)

---

Outline:

* Sample of Result
* Installation
* Training Custom Object Detector
* Testing the Models on Video

---

## A. Sample of Result
Taken using Logitech Brio 4K Pro

Image:

![image](https://user-images.githubusercontent.com/55566616/127762079-5c45bfe7-bcef-4849-b711-abb83e4f6b99.jpeg)


Video:

https://user-images.githubusercontent.com/55566616/126886984-cc2fdf0b-5752-4a7e-980e-63783f4612da.mp4


## B. Installation

### 1. Create a new Anaconda virtual environment

Open a new Terminal window and type the following command:

```
(base) C:\Users\renzo> conda create -n tensorflow pip python=3.8
```

The above will create a new virtual environment with name `tensorflow`.

### 2. Activate the Anaconda virtual environment

```
(base) C:\Users\renzo> conda activate tensorflow
```

Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

```
(tensorflow) C:\Users\renzo>
```

### 3. TensorFlow Installation:

```
(tensorflow) C:\Users\renzo> pip install --ignore-installed --upgrade tensorflow==2.4.0
```

### 4. TensorFlow Object Detection API Installation

#### (1) Create a new folder under a path of your choice and name it TensorFlow

#### (2) Downloading the TensorFlow Model Garden

* Create a new folder under a path of your choice and name it TensorFlow. (e.g., `C:\Users\renzo\TensorFlow`).
* From your Terminal `cd` into `TensorFlow` directory.
* To download the models, you can either use Git to clone the TensorFlow Models repository inside `TensorFlow` folder, or you can simply download it as a zipped file and extract its contents inside `TensorFlow` folder. To keep things consistent, in the latter case you will have to rename the extracted folder `models-master` to `models`.
* You should now have a single folder named `models` under your `TensorFlow` folder, which contains another 4 folders, such as:

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   └─ research/
```

### 5. Protobuf Installation/Compilation

* Head to the protoc releases page
* Download the latest `protoc-*-*.zip` release (e.g., `protoc-3.12.3-win64.zip` for 64-bit Windows)
* Extract the contents of the downloaded `protoc-*-*.zip` in a directory `<PATH_TO_PB>` of your choice (e.g., `C:\Program Files\Google Protobuf`)
* Add `<PATH_TO_PB>\bin` to your Path environment variable (see Environment Setup)
* In a new Terminal, `cd` into `TensorFlow/models/research/` directory and run the following command:

```python
(tensorflow) C:\Users\renzo\TensorFlow\models\research\> protoc object_detection/protos/*.proto --python_out=.
```

### 6. COCO API installation

Run the following command to install `pycocotools` with Windows support:

```python
(tensorflow) C:\Users\renzo> pip install cython
(tensorflow) C:\Users\renzo> pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### 7. Install the Object Detection API

Installation of the Object Detection API is achieved by installing the `object_detection` package. This is done by running the following commands from within `Tensorflow\models\research`:

#### (1) Install Object Detection API using the following script:

```python
# From within TensorFlow/models/research/
(tensorflow) C:\Users\renzo\TensorFlow\models\research\> cp object_detection/packages/tf2/setup.py .
(tensorflow) C:\Users\renzo\TensorFlow\models\research\> python -m pip install --use-feature=2020-resolver .
```

#### (2) Test your Installation

To test the installation, run the following script from within `Tensorflow\models\research`:

```python
# From within TensorFlow/models/research/
(tensorflow) C:\Users\renzo\TensorFlow\models\research\> python object_detection/builders/model_builder_tf2_test.py
```

## C. Training Custom Object Detector

create a new folder under TensorFlow and call it workspace. It is within the workspace that we will store all our training set-ups. Now let’s go under workspace and create another folder named training_demo. Now our directory structure should be as so:

```
TensorFlow/
├─ addons/ (Optional)
│  └─ labelImg/
├─ models/
│  ├─ community/
│  ├─ official/
│  ├─ orbit/
│  └─ research/
└─ workspace/
   └─ training_demo/
```

The `training_demo` folder shall be our training folder, which will contain all files related to our model training. It is advisable to create a separate training folder each time we wish to train on a different dataset. The typical structure for training folders is shown below.

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

### 1. Capturing Images

#### (1) Prepare Webcamera

Prepare your webcamera. Please note that the camera should have a minimum resolution of 720 dpi (1280 x 720 pixels). In my work, I use Logitech Brio 4K Pro (set at 1920 x 1080 pixels).

![https://m.media-amazon.com/images/I/810l+GhdjOL._AC_SS450_.jpg](https://user-images.githubusercontent.com/55566616/128616234-d45228ca-d80d-4a82-a223-648ada2189b6.png)

#### (2) Download `capture_image.py` Script

Download [`capture_image.py`](https://github.com/RenzoAlvaroshan/TensorFlow-2-Object-Detection-API/blob/main/Capture%20Image%20and%20Video%20files/capture_image.py) and then run the script to access your camera from computer.

```
(tensorflow) C:\Users\renzo> python capture_image.py
```

Hit <Spacebar> to capture images.

Upon finish capturing, hit <Escape> button to exit from capture_image.py.

#### (3) Capture Face Images

Capture face images from different angles.

![renzo_logitech_60_65](https://user-images.githubusercontent.com/55566616/128579829-88d9fe33-fb1d-432d-b43a-4d2ba037e853.jpeg)

![renzo_logitech_35_110](https://user-images.githubusercontent.com/55566616/128579841-8fbe10a3-447f-420b-8bb7-f1c9c706cce1.jpeg)

![renzo_logitech_15_90](https://user-images.githubusercontent.com/55566616/128579877-05c4cd1c-4f9a-4727-a6d9-c1de894bd7d3.jpeg)

To capture the images, I am using view angles within the range of 0°-180° with an interval 5° horizontally, and the range of 0°-60° with an interval 5° vertically.
In total we can have 431 pictures.

Save it into your images folder.

### 2. Preparing Dataset

Open terminal / anaconda prompt

#### (1) Install `labelImg` App

Install `labelImg` app using `pip` package manager with the following command:

```
(tensorflow) C:\Users\renzo> pip install labelImg
```

#### (2) Run `labelimg` App

Run `labelimg` using the following command:

```
(tensorflow) C:\Users\renzo> labelImg
```

#### (3) Annotate Captured Images

Annotate the faces in your image files one by one and give the face a label name (choose `Pascal/VOC` format)

I would recommend to use a macro or script for faster process of image labelling

<img width="1101" alt="Screen Shot 2021-08-07 at 05 26 27" src="https://user-images.githubusercontent.com/55566616/128579899-129a233b-df58-4619-9a43-3a3565620a06.png">

#### (4) Save Image Files

Save all you `.jpg` and `.xml` files into the same folder

### 3. Partition Dataset

To make things even tidier, let’s create a new folder `TensorFlow/scripts/preprocessing`, where we shall store scripts that we can use to preprocess our training inputs. Below is out TensorFlow directory tree structure, up to now:

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

#### (1) Download Partition Data Set Script

Download the partition data set script [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/d0e545609c5f7f49f39abc7b6a38cec3/partition_dataset.py) 👈

#### (2) Run `partition_dataset.py` Script

Then, `cd` into `TensorFlow/scripts/preprocessing` and run:

```python
(tensorflow) C:\Users\renzo\TensorFlow\scripts\preprocessing> python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1

# For example
# python partition_dataset.py -x -i C:/Users/renzo/Tensorflow/workspace/training_demo/images -r 0.1
```

This will partition our data with the ratio of 90% train data and 10% test data.

### 3. Create Label Map

TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used by  both training and detection processes.

Below we show an example label map (e.g., `label_map.pbtxt`), assuming that our dataset contains 2 labels (person name), `renzo` and `adriano`:

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

*Assuming `renzo` and `adriano` are two different persons

### 4. Create TensorFlow Records

* Click here to download the above script and save it inside `TensorFlow/scripts/preprocessing`.
* Install the pandas package:

```
(tensorflow) C:\Users\renzo\TensorFlow/scripts/preprocessing> conda install pandas  # Anaconda
                                                                                    # or
(tensorflow) C:\Users\renzo\TensorFlow/scripts/preprocessing> pip install pandas    # pip
```

Finally, cd into `TensorFlow/scripts/preprocessing` and run:

```python
# Create train data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

# Create test data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

# For example
# python generate_tfrecord.py -x C:/Users/renzo/Tensorflow/workspace/training_demo/images/train -l C:/Users/renzo/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/renzo/Documents/Tensorflow/workspace/training_demo/annotations/train.record

# python generate_tfrecord.py -x C:/Users/renzo/Tensorflow/workspace/training_demo/images/train -l C:/Users/renzo/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/renzo/Documents/Tensorflow/workspace/training_demo/annotations/test.record
```

### 5. Configure the Training Pipeline

Now that we have downloaded and extracted our pre-trained model, let’s create a directory for our training job. Under the `training_demo/models` create a new directory named `my_ssd_resnet50_v1_fpn` and copy the `training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config` file inside the newly created directory. Our `training_demo/models` directory should now look like this:

```
training_demo/
├─ ...
├─ models/
│  └─ my_ssd_resnet50_v1_fpn/
│     └─ pipeline.config
└─ ...
```

Now, let’s have a look at the changes that we shall need to apply to the `pipeline.config` file (highlighted in yellow):

<img width="697" alt="fixed_shape_resizer" src="https://user-images.githubusercontent.com/55566616/128580374-a2b03518-386e-4d23-b14b-4dca93ec59c9.png">

<img width="697" alt="y scale 10 0" src="https://user-images.githubusercontent.com/55566616/128580375-6b1777a3-8f1c-42b6-aeb2-79c3c72a0354.png">

<img width="697" alt="depth 256" src="https://user-images.githubusercontent.com/55566616/128580378-53500e3b-a164-4f4d-95dd-777e7cdfc9ce.png">

<img width="697" alt="classification weight 1 0" src="https://user-images.githubusercontent.com/55566616/128580382-f446f935-9c91-4007-9584-ddd2a3afcab3.png">

<img width="697" alt="164 replicas_to_aggregate" src="https://user-images.githubusercontent.com/55566616/128580384-2f36761d-cab7-458d-a9bc-0b8d25bf3956.png">


It is worth noting here that the changes to lines 178 to 179 above are optional. These should only be used if you installed the COCO evaluation tools, as outlined in the COCO API installation section, and you intend to run evaluation (see Evaluating the Model (Optional)).
Once the above changes have been applied to our config file, go ahead and save it.

### 6. Training the Model

Before we begin training our model, let’s go and copy the `TensorFlow/models/research/object_detection/model_main_tf2.py` script and paste it straight into our  `training_demo` folder. We will need this script in order to train our model.

Now, to initiate a new training job, open a new Terminal, `cd` inside the `training_demo` folder and run the following command:

```python
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
```

Once the training process has been initiated, you should see a series of print outs similar to the one below (plus/minus some warnings):

```
...
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.gamma
W0716 05:24:19.105542  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.gamma
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.beta
W0716 05:24:19.106541  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.beta
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_mean
W0716 05:24:19.107540  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_mean
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_variance
W0716 05:24:19.108539  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_variance
WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
W0716 05:24:19.108539  1364 util.py:151] A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
INFO:tensorflow:Step 100 per-step time 1.153s loss=0.761
I0716 05:26:55.879558  1364 model_lib_v2.py:632] Step 100 per-step time 1.153s loss=0.761
...
```

### 7. Monitor Training Job Progress using TensorBoard

A very nice feature of TensorFlow, is that it allows you to coninuously monitor and visualise a number of different training/evaluation metrics, while your model is being trained. The specific tool that allows us to do all that is Tensorboard.

   To start a new TensorBoard server, we follow the following steps:
   
* Open a new Anaconda/Command Prompt
* Activate your TensorFlow conda environment (if you have one), e.g.,:

```
activate tensorflow_gpu
```

* `cd` into the `training_demo` folder.
* Run the following command:
```
tensorboard --logdir=models/my_ssd_resnet50_v1_fpn
```

The above command will start a new TensorBoard server, which (by default) listens to port 6006 of your machine. Assuming that everything went well, you should see a print-out similar to the one below (plus/minus some warnings):

```
...
TensorBoard 2.2.2 at http://localhost:6006/ (Press `<Ctrl-C>` to quit)
```

Once this is done, go to your browser and type `http://localhost:6006/` in your address bar, following which you should be presented with a dashboard similar to the one shown below (maybe less populated if your model has just started training):


### 8. Export a Trained Model

Once your training job is complete, you need to extract the newly trained inference graph, which will be later used to perform the object detection. This can be done as follows:
   
* Copy the `TensorFlow/models/research/object_detection/exporter_main_v2.py` script and paste it straight into your `training_demo` folder.
* Now, open a Terminal, `cd` inside your `training_demo` folder, and run the following command:

```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_efficientdet_d1\pipeline.config --trained_checkpoint_dir .\models\my_efficientdet_d1\ --output_directory .\exported-models\my_model
```

After the above process has completed, you should find a new folder `my_model` under the `training_demo/exported-models` that has the following structure:

```
training_demo/
├─ ...
├─ exported-models/
│  └─ my_model/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```

## D. Testing the Models on Video

* Open the `Testing Scripts` folder from this repo

* Insert the model name with your own exported model `ssd_mobilenet_v2`

* Insert video that you want to test, for example `renzo.mp4`

* Run the following command:

```
python Object_Detection_Videos.py
```

Now you can see the result like the sample result above.

DONE…!
   
Congratulations! 👏 You have just finished creating your own custom object recognition using Tensorflow Object Detection API.

(Thank you to all people that helped me making this guidebook 🙏)

Have a nice day! 😁

~ (Renzo Adriano Alvaroshan)
