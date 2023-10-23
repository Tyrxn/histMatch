# histMatch

histMatch is a versatile tool designed for processing satellite imagery, ensuring consistent color gradients across images. By matching the color distributions of two images, it allows for a more uniform and accurate land classification. Enhancing satellite imagery by minimizing color variation, histMatch amplifies the ability to differentiate land cover types. This robust functionality finds its significance in diverse applications such as environmental management, urban planning, flood modeling, and more.

# Requirements
Python 3.6 or higher

# Installation
Clone this repository to your local machine:

git clone https://github.com/Tyrxn/histMatch

cd histMatch

# Dependencies
Set up the virtual environment and install the necessary dependencies.

python -m venv myenv

.\myenv\Scripts\activate

pip install -r requirements.txt

# Usage
python histogram_matching.py <source_img>

Select a region of interest

![image](https://github.com/Tyrxn/histMatch/assets/106474487/f885e18b-1561-447f-9ea9-1f23b352b7bf)

Press 'q' to exit GUI

# Output
![image](https://github.com/Tyrxn/histMatch/assets/106474487/27a132bb-9a8a-449d-9782-77308570a88c)


Note:

While histMatch was initially developed with QGIS in mind, its design ensures compatibility with a broad range of satellite imagery, making it a suitable tool for various geospatial applications.
