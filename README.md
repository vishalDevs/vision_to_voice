Vision to Voice: Real-Time AI Narrator (Flask Version)
Vision to Voice is an innovative AI-powered web application that transforms visual content into spoken narratives in real-time. Leveraging advanced computer vision and natural language processing, it generates descriptive captions for images, videos, and live webcam feeds, then converts these captions into speech for accessibility.

Features
Real-Time Captioning: Describe scenes from a live webcam or video in real-time using state-of-the-art models.

Multi-Source Support: Process live webcam feeds, video files, and static images through an intuitive web interface.

Dual AI Models: Uses BLIP for image captioning and Sentence-Transformers for semantic similarity filtering.

Text-to-Speech Output: Converts captions into spoken audio using pyttsx3 for an accessible experience.

Efficient Processing: Skips frames and filters similar captions for optimal resource usage and meaningful narration.

User-Friendly Interface: Modern Flask-based frontend with easy menu navigation.

Visual Output: Captioned images and video frames are saved and served via the Flask app.

Installation
Clone the repository:

text
git clone https://github.com/El-lunatico/vision-to-voice
cd vision-to-voice
Install Python dependencies (recommended: use a virtual environment):

text
pip install -r requirements.txt
(Optional) If using CUDA/GPU, ensure PyTorch is correctly installed for your platform.

Running the Application
Start the Flask app:

text
python app.py
This launches the web server on http://localhost:5000. Open the URL in your browser to access the interface.

Usage
Live Webcam: Start real-time scene narration directly from your computer’s webcam.

Video File: Upload a video to generate and hear scene captions as the video plays.

Single Image: Upload an image and receive a spoken and visual caption.

Requirements
Python 3.8+

Flask

OpenCV (opencv-python)

Torch and torchvision

transformers (transformers)

sentence-transformers

pyttsx3

PIL

numpy

Refer to requirements.txt for full details.

Author and Credits
Developed by El-lunatico and contributors.

Core Logic: Pushkar Singh (Computer Vision, Flask integration)

AI: Vishal Sharma (NLP, Model Optimization)

Frontend: Aashutosh Kumar (User Interface Design)

About
Vision to Voice is designed to provide an accessible audio description of visual content for improved inclusivity. Built with cutting-edge AI tools, it offers both real-time and batch processing in a simple, user-friendly package.
