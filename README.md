# YOLOv3 with TensorRT engine

This package contains the yolo_trt_node that performs object detection with YOLOv3 using NVIDIA's TensorRT engine


![Video_Result2](docs/results.gif)

---
## Setting up the environment

### Install dependencies

### Current Environment:

- Jetson Xavier AGX
- ROS Melodic
- Ubuntu 18.04
- Jetpack 4.5.1
- TensorRT 7+

#### Dependencies:

- OpenCV 4.2.0
- numpy 1.15.1
- Protobuf 3.8.0 -> Not necessary for deployment
- Pycuda 2019.1.2
- onnx 1.4.1 (depends on Protobuf) -> Not necessary for deployment

### Install all dependencies with below commands

```
Install pycuda (takes a while)
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/dependencies
$ ./install_pycuda.sh

Install Protobuf (takes a while)
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/dependencies
$ ./install_protobuf-3.8.0.sh

Install onnx (depends on Protobuf above)
$ sudo pip3 install onnx==1.4.1
```

### Build vision_opencv from source
ROS Melodic depends on OpenCV 3 but Jetpack 4.5.1 depends on OpenCV 4. Thus, the ROS packages used that depend on OpenCV must be built from source

Clone the vision_opencv package from the 'melodic' branch  https://github.com/ros-perception/vision_opencv/tree/melodic
```
git clone -b melodic --single-branch git@github.com:ros-perception/vision_opencv.git
```
A few modifications to the package must be made to build it with OpenCV 4:
1. Add set (CMAKE_CXX_STANDARD 11) to your top level cv_bridge cmake
2. In cv_bridge/src CMakeLists.txt line 35 change to if (OpenCV_VERSION_MAJOR VERSION_EQUAL 4)
3. In cv_bridge/src/module_opencv3.cpp change signature of function
   ```
   UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const
   ```
   to
   ```
   UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const
   ```
4. Still in cv_bridge/src/module_opencv3.cpp change signature of function
   ```
   bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
   ```
   to
   ```
   bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const
   ```


---
## Setting up the package

### 1. Clone project into catkin_ws and build it

``` 
$ cd ~/catkin_ws && catkin build 
$ source devel/setup.bash
```

### 2. Make libyolo_layer.so

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/plugins
$ make
```

This will generate a libyolo_layer.so file.

### 3. Place your yolo.weights and yolo.cfg file in the yolo folder

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/yolo
```
**Please name the yolov3.weights and yolov3.cfg file as follows:**
- yolov3.weights
- yolov3.cfg

Run the conversion script to convert to TensorRT engine file

```
$ ./convert_yolo_trt
```

- Input the appropriate arguments:
   - *input_shape* is the input shape of the yolo network 
   - *max_batch_size* is the maximum batch size of the TensorRT engine. The resulting engine will be able to infer images with a batch size smaller or equal than *max_batch_size*. For example, if *max_batch_szie* is set to 8, the resulting engine will be able to infer images with a batch size of 1, 2, 4 and 8. A runtime batch size equal to the *max_batch_size* will yield optimal performances. Smaller runtime batch sizes will work but with a sub-optimal framerate. 
   If you are sure of the batch size you will use at runtime, set *max_batch_size* to this value. This will yield optimal performances. If you are unsure about your runtime batch size, set *max_batch_size* to a large power of 2. 
- This conversion might take a while
- The optimised TensorRT engine would now be saved as yolov3-416.trt (if 416 is the input shape)

If convert_yolo_trt script doesn't work, create the weights manually:

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/yolo
```
**Please name the yolov3.weights and yolov3.cfg file as follows:**
- yolov3-416.weights
- yolov3-416.cfg
(replace 416 with your network input shape: '288', '416' or '608')

```
For yolov3:
$ python3 yolo_to_onnx.py -m yolov3-<input_shape> -c <category_num>

For yolov3-tiny:
$ python3 yolo_to_onnx.py -m yolov3_tiny-<input_shape> -c <category_num> --verbose
```

This step should take around a minute (depending on the size of the weight file).Next: 

```
For yolov3:
$ python3 onnx_to_tensorrt.py -m yolov3-<input_shape> -c <category_num> -b <max_batch_size>

For yolov3-tiny:
$ python3 onnx_to_tensorrt.py -m yolov3_tiny-<input_shape> -c <category_num> -b <max_batch_size>
```

This step should take a few minutes. Feel free to grab a coffee while the engine is being created.

### 4. Change the class labels

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/utils
$ vim yolo_classes.py
```

- Change the class labels to suit your model

### 5. Change the *.yaml parameters

```
$ cd ${HOME}/catkin_ws/src/yolo_trt_ros/config
```

- `ros.yaml` : change the camera topic names. `yolov3_trt.launch` only subscribes to the front camera topic. `yolov3_trt_batch.launch` subscribes to all 4 camera topics.
- `ros.yaml` : change resolution of cameras. If resolution unknown, enter `2**26`

- `yolov3.yaml` : change parameters accordingly:
   - str model = 'yolov3' or 'yolov3_tiny' 
   - int input_shape = '288' or '416' or '608'
   - int category_num = 8
   - int batch_size = 1 or 4
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
