## AAI Group-1 🤖

A music genre classifier model

![front end one](./img/front-end-1.png?raw=true "Front end one")
![front end two](./img/front-end-2.png?raw=true "Front end two")

### Authors

* Shuaib Shahbaz
* Alexander Feetham
* Ethan Sheridan-Smith
* Alex Roussel-Smith

## Built with 🔨

* [Python 3.9.*](https://www.python.org/downloads/release/python-3916/)
* [Tensorflow](https://www.tensorflow.org/)
* [FFMPEG](https://ffmpeg.org/)
* [streamlite](https://streamlit.io/)

## Prerequisites 🔧
Make sure you have python 3.9 and FFMPEG installed on PATH environment

## Install
### Clone the repo
If you're viewing this via github launch a terminal in your file directory and enter the command
```
https://github.com/ethansheridansmith/AAI && cd AAI
```
### Setup project
Create a python virtual environment
```
python -m venv env
```
Now activate our env
```
.\env\Scripts\activate
```
Finally just install the requirements and you're ready to go!
```
pip install -r requirements.txt
```

## Run 🚀
Make sure the virtual environment is running or requirements have been installed
```
streamlite run .\src\app.py
```
Then navigate to [localhost](http://localhost:8501/) on the specified port