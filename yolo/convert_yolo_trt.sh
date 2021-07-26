echo "Are you using YOLOv3-tiny ? [y/n]"
read tiny

if [ $tiny == 'y' ]
then
    yolo_type="_tiny"
else
    yolo_type=""
fi

echo "What is the input shape? Input 288, 416 or 608"
read input_shape

if [ $tiny == 'y' ]
then
    if [[ $input_shape == 288 ]]
    then
        echo "Creating yolov3_tiny-288.cfg and yolov3_tiny-288.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=288/' | sed -e '8s/height=608/height=288/' > yolov3_tiny-288.cfg
        ln -sf yolov3.weights yolov3_tiny-288.weights
    fi
    if [[ $input_shape == 416 ]]
    then
        echo "Creating yolov3_tiny-416.cfg and yolov3_tiny-416.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=416/' | sed -e '8s/height=608/height=416/' > yolov3_tiny-416.cfg
        ln -sf yolov3.weights yolov3_tiny-416.weights
    fi
    if [[ $input_shape == 608 ]]
    then
        echo "Creating yolov3_tiny-608.cfg and yolov3_tiny-608.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' > yolov3_tiny-608.cfg
        ln -sf yolov3.weights yolov3_tiny-608.weights
    fi
else
    if [[ $input_shape == 288 ]]
    then
        echo "Creating yolov3-288.cfg and yolov3-288.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=288/' | sed -e '8s/height=608/height=288/' > yolov3-288.cfg
        ln -sf yolov3.weights yolov3-288.weights
    fi
    if [[ $input_shape == 416 ]]
    then
        echo "Creating yolov3-416.cfg and yolov3-416.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' | sed -e '7s/width=608/width=416/' | sed -e '8s/height=608/height=416/' > yolov3-416.cfg
        ln -sf yolov3.weights yolov3-416.weights
    fi
    if [[ $input_shape == 608 ]]
    then
        echo "Creating yolov3-608.cfg and yolov3-608.weights"
        cat yolov3.cfg | sed -e '2s/batch=64/batch=1/' > yolov3-608.cfg
        ln -sf yolov3.weights yolov3-608.weights
    fi
fi

echo "How many object categories can your model detect?"
read category_num
model_name="yolov3${yolo_type}-${input_shape}"

echo "What maximum batch size does your engine need? Input a power of 2 (1, 4, 8, 16, etc)"
read max_batch_size

# convert from yolo to onnx
python3 yolo_to_onnx.py -m $model_name -c $category_num 

echo "Done converting to .onnx"
echo "..."
echo "Now converting to .trt"

# convert from onnx to trt
python3 onnx_to_tensorrt.py -m $model_name -c $category_num -b max_batch_size --verbose

echo "Conversion from yolo to trt done!"
echo "..."
echo "Now converting to .trt"

# convert from onnx to trt
python3 onnx_to_tensorrt.py -m $model_name -c $category_num --verbose

echo "Conversion from yolo to trt done!"
