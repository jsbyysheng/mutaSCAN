# mutaSCAN

This repository is for **High-throughput and integrated CRISPR/Cas12a-based molecular diagnosis using a deep learning-enabled microfluidic system**.

## Hardware Requirements
- Raspberry Pi 4B with 8G RAM
- Arducam IMX519
- JJC Neutral Density Filter
- JUYING DAM0404D

To summarize, JJC Neutral Density Filter is used to reduce the intensity of light entering the Arducam IMX519. Arducam IMX519 connected to Raspberry Pi is used to capture images of the microfluidic chip. 8G RAM is required to run the deep learning model. JUYING DAM0404D is the industrial relay control board which can be used to control UV light. More details can be found in the paper.

## Software Environment
The recommanded IDE is Visual Studio Code. The training and inference of the deep learning model are done using Detectron2. Model was trained in Ubuntu 20.04 with RTX3090Ti and then used for inference on Raspberry Pi with Debian Bullseye aarch64.

### S0 Clone
```bash
git clone https://github.com/jsbyysheng/mutaSCAN.git
cd mutaSCAN
```

### S1 Hardware
Codes for hardware control are written in Python and running in system python environment. You can install Arducam IMX519 driver followed the `documents/arducam-imx519-start-guide.pdf`. Using `!libcamera-still -t 5 -n -o cache.jpg` to check if the camera works properly.

```bash
pip install numpy matplotlib pillow pymodbus pyserial tqdm
sudo apt install python3-pyqt5 qttools5-dev-tools
```

### S2 Deep Learning
Install [miniforge3](https://github.com/conda-forge/miniforge) for conda environment management. Create a new conda environment named `detectron2` with Python 3.10. The deep learning dependencies are listed in the `environment.yml` file.
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
mamba env create --name detectron2 --file=environment.yml
mamba activate detectron2
tar -zxvf model/weights.tar.gz -C model
```

## File Structure
`0_sample_data.ipynb`, Device: Raspberry Pi 4B, Environment: system python3. This jupyter notebook can be used to sample data.

`1_preprocess.ipynb`, Device: Workstation, Environment: conda_env detectron2. This jupyter notebook can be used to preprocess sample data.

`2_convert_to_coco.ipynb`, Device: Workstation, Environment: conda_env detectron2. This jupyter notebook can be used to convert sample data, which has been preprocessed and labelled by [labelme](https://github.com/labelmeai/labelme), to coco dataset.

`3_deeplearning.ipynb`, Device: Workstation, Environment: conda_env detectron2. This jupyter notebook can be used to train model.

`inference_server.py`, Device: Raspberry Pi 4B, Environment: conda_env detectron2. This python script can be used to run inference server. You can execute `python inference_server.py` to start the server. To test the inference server, you can use this code:
```python
import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "img_path": 'dataset/sample_preprocessed.png',
}
response = requests.post(url, json=data)
if response.status_code == 200:
    prediction = response.json()
    print(prediction)
else:
    print(f'API Request Failed with Status Code: {response.status_code}')
    print(f'Response Content: {response.text}')
```

`labelme2coco.py` The python script from [Tony607](https://github.com/Tony607/labelme2coco) can be used to convert labelme json file to coco json file.

`utility.py` The python script contains some utility functions.

`Folder: focus` The folder contains the camera control library.

`Folder: model` The folder contains the trained model weights which can be used in `inference_server.py` directly.

`Folder: documents` The folder contains the documents for `Arducam IMX519`.

`Folder: datasets` The folder contains some samples for validation.
