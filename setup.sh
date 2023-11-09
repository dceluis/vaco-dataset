pip install -r requirements.txt

if [ ! -d "JSON2YOLO" ]; then
  git clone https://github.com/ultralytics/JSON2YOLO.git --depth 1
fi

pip install -r JSON2YOLO/requirements.txt
