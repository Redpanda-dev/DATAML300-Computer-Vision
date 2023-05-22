# If the TC315 or TC303 are missing python packages try the following.

# TO USE UNPACK python/Task_2-3/weights/VGG_VOC0712_SSD_300x300_iter_120000

if `python --version` returns python3.9 do the following

# Create a virtual environment
- `python -m venv venv`
- `./venv/Scripts/activate`

# Install missing packages
- `pip install tensorflow==2.5 scikit-image opencv-python matplotlib`


elif `python --version` returns python3.7 do the following

# Create a virtual environment
- `python -m venv venv`
- `./venv/Scripts/activate`
# Install missing packages
- `pip install tensorflow==2.3 scikit-image opencv-python matplotlib`

else:

# Ask for help
