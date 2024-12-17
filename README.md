# License Plate OCR
A part of a smart-parking system with automatic license plate detection, and license plate number recognition.

This system combines a custom YOLO model trained to detect (Pakistani) license plates. After detecting the license plate, once it leaves frame, the best image of the license plate taken alongside the determined text are displayed in top left of video feed.


## Demo

![Demo GIF](https://github.com/FaizaanAlFaisal/License-Plate-OCR/blob/main/assets/PlateOCR-Demo.gif)


## Usage

Setup the environment and install requirements. Ensure PyTorch with cuda support is enabled.

Within the `tracking.py` file, change the parameters of object being created as needed, and modify .env with path to video feed or URL of live feed. 

Run the file with:

```python
python tracking.py
```
