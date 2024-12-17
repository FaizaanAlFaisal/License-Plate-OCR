# Smart-Parking
Smart parking system with automatic license plate detection, and license plate number recognition.

This system combines a custom YOLO model trained to detect (Pakistani) license plates. After detecting the license plate, once it leaves frame, the best image of the license plate alongside the determined text are shown in top left.


_____


## Demo

![Demo GIF](.\assets\PlateOCR-Demo.gif)

_____

## Usage

Setup the environment and install requirements. Ensure PyTorch with cuda support is enabled.

Within the `tracking.py` file, change the parameters of object being created as needed, and modify .env with path to video feed or URL of live feed. 

Run the file with:

```python
python tracking.py
```