# YOLOv3 with TensorRT engine

This package contains the yolo_trt_node that performs object detection with YOLOv3 using NVIDIA's TensorRT engine

---
## Setting up the environment

**Download latest weights**: check latest folder in [TensorRT Weights History](https://drive.google.com/drive/u/1/folders/1APAwXVrVQY7cb_p0FCEzutJ3G2x3OBnL) and place content in folder `yolo/`.


### Install dependencies

### Current Environment:

- Jetson Xavier AGX
- ROS Melodic
- Ubuntu 18.04
- Jetpack 4.5
- TensorRT 7+

#### Dependencies *to generate weights*:
Dependencies to have on the machine that generates tensorRT weights. This is not necessarily the machine that will run the network but must be of the same type. Example: One Jetson Xavier can create the trt weights that then will be deployed on other Jetson Xavier.

- OpenCV 4.1.1
- numpy 1.19.1
- Protobuf 3.8.0
- Pycuda 2019.1.2
- onnx 1.4.1 (depends on Protobuf)

```
Install pycuda (takes a while)
$ cd yolo_trt_ros/dependencies
$ ./install_pycuda.sh

Install Protobuf (takes a while)
$ cd yolo_trt_ros/dependencies
$ ./install_protobuf-3.8.0.sh

Install onnx (depends on Protobuf above)
$ sudo pip3 install onnx==1.4.1
```

#### Dependencies *to run network* (NO weights generation):
Dependencies to have on the machine that uses tensorRT weights to detect objects.

- OpenCV 4.1.1
- Pycuda 2019.1.2

```
Install pycuda (takes a while)
$ cd yolo_trt_ros/dependencies
$ ./install_pycuda.sh
```

### Build vision_opencv from source
ROS Melodic depends on OpenCV 3 but Jetpack 4.5 depends on OpenCV 4. Thus, the ROS packages used that depend on OpenCV must be built from source

```
git clone -b feature/darpa --single-branch git@github.com:leggedrobotics/vision_opencv.git
```

---
## Weights generation


### 1. Make libyolo_layer.so

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/plugins
$ make
```

This will generate a libyolo_layer.so file.

### 2. Place your yolo.weights and yolo.cfg file in the yolo folder

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/yolo
```
** Please name the yolov3.weights and yolov3.cfg file as follows:
- yolov3.weights
- yolov3.cfg

**Set the maximum branch size you want to use** in files [yolo_to_onnx.yaml](./yolo/yolo_to_onnx.yaml) and [onnx_to_tensorrt.yaml](./yolo/onnx_to_tensorrt.yaml), change varialbe `MAX_BATCH_SIZE`.

Run the conversion script to convert to TensorRT engine file

```
$ ./convert_yolo_trt.sh
```

- Input the appropriate arguments
- This conversion might take awhile
- The optimised TensorRT engine would now be saved as yolov3-416.trt / yolov3-416.trt

If convert_yolo_trt script doesn't work, create the weights manually:

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/yolo
```

```
For yolov3:
$ python3 yolo_to_onnx.py -m yolov3-<input_shape> -c <category_num>

For yolov3-tiny:
$ python3 yolo_to_onnx.py -m yolov3_tiny-<input_shape> -c <category_num> --verbose
```

This step should take around a minute (depending on the size of the weight file).Next:

```
For yolov3:
$ python3 onnx_to_tensorrt.py -m yolov3-<input_shape> -c <category_num>

For yolov3-tiny:
$ python3 onnx_to_tensorrt.py -m yolov3_tiny-<input_shape> -c <category_num>
```

This step should take a few minutes. Feel free to grab a coffee while the engine is being created.

### 3. Change the class labels (if needed)

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/utils
$ vim yolo_classes.py
```

- Change the class labels to suit your model

### 4. Change the *.yaml parameters

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/config
```

- `ros.yaml` : change the camera topic names. `yolov3_trt.launch` only subscribes to the front camera topic. `yolov3_trt_batch.launch` subscribes to all 4 camera topics.
- `ros.yaml` : change resolution of cameras. If resolution unknown, enter `2**26`

- `yolov3.yaml` : change parameters accordingly:
   - str model = 'yolov3' or 'yolov3_tiny'
   - int input_shape = '288' or '416' or '608'
   - int category_num = 8
   - int batch_size = 1 or 4 **<- important parameter!**
   - double confidence_threshold = 0.3

### 6. Change the rosbag
<em>OPTIONAL: if running on rosbag</em>

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_node/launch
```
- `rosbag.launch` : change rosbag path
---
## Using the package

### Running the package

Note: Run the launch files separately in different terminals

### 1. Run the yolo detector
```
# For YOLOv3 (single input)
$ roslaunch yolo_trt_ros yolov3_trt.launch

# For YOLOv3 batch (multiple input)
$ roslaunch yolo_trt_ros yolov3_trt_batch.launch

```
If using a rosbag, in a split terminal:
```
$ source devel/setup.bash
$ roslaunch yolo_trt_node rosbag.launch
```


### 2. For maximum performance
```
sudo -H pip install -U jetson-stats
```
In a seperate terminal:
```
$ jtop
```
* Press 5 to access the control tab of the Jetson:
   * Increase fan speed by pressing 'p'. Reduce fan speed by pressing 'm'.
   * Overclock GPU by pressing 's'.
   * Select 'MAXN' mode by clicking on it.


* These commands are found/referred in this [repo](https://github.com/rbonghi/jetson_stats/wiki/jtop)
* Please ensure the jetson device is cooled appropriately to prevent overheating



---
## Results obtained

### Inference Results
#### Single Camera Input

   | Model    | Hardware |    FPS    |  Inference Time (ms)  |
   |:---------|:--:|:---------:|:----------------:|
   | yolov3-416| Xavier AGX | 41.0 | 0.024 |
   | yolov3_tiny-416| Xavier AGX | 102.6 | 0.0097 |

---
## Licenses and References

### 1. TensorRT samples from [jkjung-avt](https://github.com/jkjung-avt/)

Many thanks for his project with tensorrt samples. I have referenced his source code and adapted it to ROS for robotics applications.

I also used the pycuda and protobuf installation script from his project

Those codes are under [MIT License](https://github.com/jkjung-avt/tensorrt_demos/blob/master/LICENSE)

### 2. yolo_trt_ros from [indra4837](https://github.com/indra4837/yolo_trt_ros)

Many thanks to his work on creating most of what this package is built upon! The package is forked from his repository.

Those codes are under [MIT License](https://github.com/indra4837/yolo_trt_ros/blob/master/LICENSE)
