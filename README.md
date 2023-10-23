# histMatch

histMatch is a versatile tool designed for processing satellite imagery, ensuring consistent color gradients across images. By matching the color distributions of two images, it allows for a more uniform and accurate land classification. Enhancing satellite imagery by minimizing color variation, histMatch amplifies the ability to differentiate land cover types. This robust functionality finds its significance in diverse applications such as environmental management, urban planning, flood modeling, and more.

# Requirements
Python 3.6 or higher
# Dependencies
Make sure to install the required dependencies by setting up a virtual environment and using the provided requirements.txt:

python3 -m venv histMatch_env
source histMatch_env/bin/activate  # On Windows, use: histMatch_env\Scripts\activate
pip install -r requirements.txt

# Installation
Clone this repository to your local machine:

git clone https://github.com/Tyrxn/histMatch
cd histMatch

Set up the virtual environment and install the necessary dependencies as mentioned in the "Dependencies" section.

Once dependencies are installed, you're ready to use histMatch with satellite imagery of your choice.

# Usage
python histogram_matching.py <source_img>

Note
While histMatch was initially developed with QGIS in mind, its design ensures compatibility with a broad range of satellite imagery, making it a suitable tool for various geospatial applications.
