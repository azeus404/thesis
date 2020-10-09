#!/usr/bin/env bash
echo '[**] Run all models [**]'
python3 base2.py -m ResNet50
python3 base2.py -m ResNet50V2
python3 base2.py -m ResNet101V2
python3 base2.py -m ResNet152V2
python3 base2.py -m InceptionV3
python3 base2.py -m Xception
python3 base2.py -m VGG16
python3 base2.py -m VGG19
python3 base2.py -m MobileNet
python3 base2.py -m MobileNetV2
python3 base2.py -m EfficientNetB7
python3 base2.py -m InceptionResNetV2
python3 base2.py -m DenseNet121
python3 base2.py -m DenseNet169
python3 base2.py -m NASNetLarge
echo '[*] done [*]'
