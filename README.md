
# TEXT-RECOGNITION-USING-PYTHON



Text Recognition from images is an active research 
area which attempts to develop a computer application 
with the ability to automatically read texts from images.
The project is based on the same idea. Scene based text detection,
document scanning, 2d image scanning are the main highlights.

Language used-PYTHON 

Modules Used-Opencv,Pytesseract,Easyocr,Tkinter


## Dependencies
Python-tesseract is a wrapper for Google’s Tesseract-OCR Engine(https://github.com/tesseract-ocr/tesseract).
It can read all image types supported by the Pillow and Leptonica 
imaging libraries, 
including jpeg, png, gif, bmp, tiff, and others. In addition it can also 
extract text from images and write it into another file.
```bash
pip install pytesseract
```
Specify the tesseract.exe directory in the code to the cmd.

Here it is in my case-
p.pytesseract.tesseract_cmd=r'C:\Users\hp\AppData\Local\Tesseract-OCR\tesseract.exe'

For Windows, please install torch and torchvision first by following the official instruction here https://pytorch.org. On pytorch website, be sure to select the right CUDA version you have. 
If you intend to run on CPU mode only, select CUDA = None
```bash
pip install easyocr
```
PIL is the Python Imaging Library which 
adds image processing capabilities to your Python interpreter.
```bash
pip install Pillow
```
## Deployment

To deploy this project first clone the repository 
using git command or by downloading the zip file.
Now forward to the downloaded directory.Run the runner.py file.

```bash
  python runner.py
```
 After the tkinter ui loads select the option which best describes the image type
 - scene detection(for license plate recognition,sign boards,name plates)
 - document scanner(for images of pdf type)
 - 2d image scanner(for skewed images)
 
 Then select the image from the image browser and press Enter. This will load the image and the results.
  
## Acknowledgements
Thanks to the youtube channel by-[
Nicholas Renotte](https://www.youtube.com/channel/UCHXa4OpASJEwrHrLeIzw7Yg)


Thanks to the youtube channel-[
Murtaza's Workshop-Robotics and AI
](https://www.youtube.com/watch?v=ON_JubFRw8M)

Article on ocr-[
Nanonets
](https://nanonets.com/blog/ocr-with-tesseract/)


  
