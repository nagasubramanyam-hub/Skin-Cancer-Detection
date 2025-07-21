
from setuptools import setup

setup(
    name="skin-cancer-detection-web",
    version="1.0.0",
    install_requires=[
        "Flask==2.3.3",
        "Flask-Cors==4.0.0",
        "tensorflow==2.16.1",
        "numpy>=1.26.0",
        "pillow==9.5.0",
        "matplotlib==3.8.0",
        "scikit-learn==1.3.2",
        "opencv-python-headless==4.7.0.72",
        "gdown==4.6.0",
        "Werkzeug==2.3.7",
        "gunicorn==21.2.0"
    ],
)