"""
Vision to Voice: A Real-Time AI Narrator (Flask Version)

This Flask application integrates the original Vision to Voice script without modifying its code.
It provides a web interface to trigger the original functions for describing scenes from a live
webcam, a video file, or a static image in real-time.

The core logic remains unchanged, leveraging the BLIP model for image captioning and
Sentence-Transformers for semantic similarity. OpenCV is used for video/image processing,
and pyttsx3 for text-to-speech narration. Visual output is saved to disk and served via Flask.

Author: [https://github.com/El-lunatico]
Date: 29/07/2025
Flask Adaptation: 01/09/2025
"""
import os
import threading
from flask import Flask, Response, render_template_string, request, redirect, url_for, send_file
from queue import Queue
from io import BytesIO
from base64 import b64encode

# --- ORIGINAL CODE (UNCHANGED) ---
import os
import threading
from typing import Union, Optional

import cv2
import numpy as np
import pyttsx3
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- CONFIGURATION ---
# You can tweak these values to change the application's behavior.

# The index of the webcam to use. 0 is usually the default built-in webcam.
WEBCAM_INDEX: int = 0
# How many frames to skip between processing. Higher values save resources.
FRAME_SKIP: int = 30
# The percentage of pixel change required to trigger a new caption.
# A lower value makes it more sensitive to small movements.
CHANGE_THRESHOLD_RATIO: float = 0.09
# The semantic similarity score above which a new caption is considered
# "the same" as the last one and will not be spoken.
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.95

# --- MODEL AND DEVICE SETUP ---
def setup_models():
    """
    Loads all required AI models and sets up the computation device.
    """
    print("Loading models... This may take a moment. 🤖")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the BLIP model for image captioning
    print(" -> Loading Image Captioning model (BLIP)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model.to(device)

    # Load the Sentence Transformer for semantic similarity
    print(" -> Loading Sentence Similarity model (all-MiniLM-L6-v2)...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    similarity_model.to(device)

    print(f"Models loaded successfully. Using device: {device} 🚀")
    return device, processor, caption_model, similarity_model

# --- CORE FUNCTIONS ---
def speak(text: str) -> None:
    """
    Initializes a new TTS engine, speaks the given text, and shuts down.
    This is run in a separate thread to prevent the main GUI from freezing.

    Args:
        text: The string to be spoken.
    """
    if not text:
        return
    try:
        # Each thread needs its own pyttsx3 engine instance
        engine = pyttsx3.init()
        print(f"🔊 Speaking: '{text}'")
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error in Text-to-Speech engine: {e}")

def generate_caption(rgb_image: np.ndarray, processor, model, device) -> str:
    """
    Generates a descriptive caption for a single image.

    Args:
        rgb_image: The input image as a NumPy array in RGB format.
        processor: The BLIP processor.
        model: The BLIP model.
        device: The torch device (CPU or CUDA).

    Returns:
        The generated caption as a string.
    """
    # Convert numpy array to PIL Image
    image = Image.fromarray(rgb_image).convert("RGB")
    
    # Process the image and generate caption
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
        
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption

def process_image(image_path: str, processor, caption_model, device) -> None:
    """
    Loads, captions, and displays a single static image.

    Args:
        image_path: The file path to the image.
        processor: The BLIP processor.
        caption_model: The BLIP model.
        device: The torch device.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from path: {image_path}")
        return

    print("\n🖼️  Processing single image...")
    # Convert from BGR (OpenCV's default) to RGB for the model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    caption = generate_caption(image_rgb, processor, caption_model, device)
    print(f"✅ Generated Caption: '{caption}'")

    # Start speaking in a background thread
    threading.Thread(target=speak, args=(caption,), daemon=True).start()

    # Display the image with the caption
    cv2.putText(image, f"Caption: {caption}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Image Caption (Press any key to close)', image)
    
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_source: Union[str, int], models) -> None:
    """
    Processes a video source (webcam or file) frame by frame.

    Args:
        video_source: The path to a video file or the index of a webcam.
        models: A tuple containing the device, processor, and models.
    """
    device, processor, caption_model, similarity_model = models
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        return

    # State variables for the processing loop
    prev_frame_gray: Optional[np.ndarray] = None
    last_caption_embedding: Optional[torch.Tensor] = None
    last_spoken_caption: str = "Initializing..."
    frame_count: int = 0

    print("\n▶️  Starting video processing... Press 'q' on the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            break

        frame_count += 1
        # Process frames only at a specific interval (FRAME_SKIP)
        if frame_count % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for significant change in the scene
            significant_change = True
            if prev_frame_gray is not None:
                diff = cv2.absdiff(gray, prev_frame_gray)
                change_ratio = np.count_nonzero(diff) / diff.size
                if change_ratio < CHANGE_THRESHOLD_RATIO:
                    significant_change = False
            
            if significant_change:
                prev_frame_gray = gray
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Generate a new caption for the current frame
                current_caption = generate_caption(frame_rgb, processor, caption_model, device)
                
                # 3. Check for semantic similarity to the last spoken caption
                is_new_meaning = True
                if last_caption_embedding is not None and current_caption:
                    current_embedding = similarity_model.encode(current_caption, convert_to_tensor=True, device=device)
                    cosine_score = util.cos_sim(current_embedding, last_caption_embedding)[0][0].item()
                    
                    if cosine_score > SEMANTIC_SIMILARITY_THRESHOLD:
                        is_new_meaning = False
                
                # 4. If the caption is new and meaningful, speak it
                if is_new_meaning and current_caption:
                    last_spoken_caption = current_caption
                    # Use a thread for speaking to avoid blocking the video feed
                    threading.Thread(target=speak, args=(last_spoken_caption,), daemon=True).start()
                    last_caption_embedding = similarity_model.encode(last_spoken_caption, convert_to_tensor=True, device=device)

        # Display the frame with the most recent caption
        display_frame = frame.copy()
        cv2.putText(display_frame, f"AI: {last_spoken_caption}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Vision to Voice (Press "q" to quit)', display_frame)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting by user command...")
            break

    print("Processing finished. Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to run the application. Displays a menu for the user.
    """
    # Load models once at the start
    models = setup_models()
    
    while True:
        print("\n" + "="*30)
        print("    Vision to Voice Menu")
        print("="*30)
        print("  1: 🎙️  Use Live Webcam")
        print("  2: 🎬 Use a Video File")
        print("  3: 🖼️  Process a Single Image")
        print("  4: 🚪 Exit")
        print("-"*30)
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            process_video(WEBCAM_INDEX, models)
        elif choice == '2':
            file_path = input("Please enter the full path to your video file: ")
            if os.path.exists(file_path):
                process_video(file_path, models)
            else:
                print("\n❌ Error: File not found. Please check the path and try again.\n")
        elif choice == '3':
            file_path = input("Please enter the full path to your image file: ")
            if os.path.exists(file_path):
                process_image(file_path, models[0], models[1], models[2]) # Pass only needed models
            else:
                print("\n❌ Error: File not found. Please check the path and try again.\n")
        elif choice == '4':
            print("Exiting program. Goodbye! 👋")
            break
        else:
            print("\n⚠️ Invalid choice. Please enter 1, 2, 3, or 4.\n")

# --- END OF ORIGINAL CODE ---

# --- FLASK APP SETUP ---
app = Flask(__name__)

# Global variables
models = None
stop_stream = False
output_queue = Queue()  # To store captions and frames/images for web display
output_dir = 'static/outputs'
os.makedirs(output_dir, exist_ok=True)

# --- MODIFIED CORE FUNCTIONS FOR FLASK ---
def process_image_flask(image_path: str, processor, caption_model, device) -> None:
    """
    Wrapper for process_image to save output to disk and store caption.
    """
    global output_queue
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from path: {image_path}")
        output_queue.put(("error", "Error: Could not load image."))
        return

    print("\n🖼️  Processing single image...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    caption = generate_caption(image_rgb, processor, caption_model, device)
    print(f"✅ Generated Caption: '{caption}'")

    threading.Thread(target=speak, args=(caption,), daemon=True).start()

    # Add caption to image and save to disk
    cv2.putText(image, f"Caption: {caption}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    output_path = os.path.join(output_dir, 'output_image.jpg')
    cv2.imwrite(output_path, image)
    output_queue.put(("image", caption, output_path))

def process_video_flask(video_source: Union[str, int], models) -> None:
    """
    Wrapper for process_video to save frames to disk and store captions.
    """
    global stop_stream, output_queue
    device, processor, caption_model, similarity_model = models
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        output_queue.put(("error", "Error: Could not open video source."))
        return

    prev_frame_gray: Optional[np.ndarray] = None
    last_caption_embedding: Optional[torch.Tensor] = None
    last_spoken_caption: str = "Initializing..."
    frame_count: int = 0

    print("\n▶️  Starting video processing...")

    while not stop_stream:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            output_queue.put(("end", "End of video file reached."))
            break

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            significant_change = True
            if prev_frame_gray is not None:
                diff = cv2.absdiff(gray, prev_frame_gray)
                change_ratio = np.count_nonzero(diff) / diff.size
                if change_ratio < CHANGE_THRESHOLD_RATIO:
                    significant_change = False
            
            if significant_change:
                prev_frame_gray = gray
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_caption = generate_caption(frame_rgb, processor, caption_model, device)
                is_new_meaning = True
                if last_caption_embedding is not None and current_caption:
                    current_embedding = similarity_model.encode(current_caption, convert_to_tensor=True, device=device)
                    cosine_score = util.cos_sim(current_embedding, last_caption_embedding)[0][0].item()
                    if cosine_score > SEMANTIC_SIMILARITY_THRESHOLD:
                        is_new_meaning = False
                
                if is_new_meaning and current_caption:
                    last_spoken_caption = current_caption
                    threading.Thread(target=speak, args=(last_spoken_caption,), daemon=True).start()
                    last_caption_embedding = similarity_model.encode(last_spoken_caption, convert_to_tensor=True, device=device)

        # Save frame with caption
        display_frame = frame.copy()
        cv2.putText(display_frame, f"AI: {last_spoken_caption}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(output_path, display_frame)
        output_queue.put(("frame", last_spoken_caption, output_path))

    print("Processing finished. Cleaning up...")
    cap.release()

# --- FLASK ROUTES ---
@app.route('/')
def index():
    """Renders the main page with navbar, menu, about, features, team, and footer."""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vision to Voice</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Arial', sans-serif;
            }

            body {
                background: url('https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?q=80&w=2070&auto=format&fit=crop') no-repeat center center fixed;
                background-size: cover;
                color: #fff;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                position: relative;
            }

            /* Overlay for better text readability */
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.4);
                z-index: -1;
            }

            /* Fallback background */
            @supports not (background: url('')) {
                body {
                    background: linear-gradient(135deg, #1e3c72, #2a5298);
                }
            }

            /* Navbar */
            .navbar {
                position: fixed;
                top: 0;
                width: 100%;
                background: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(10px);
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 1000;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            }

            .navbar h1 {
                font-size: 1.8rem;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }

            .navbar ul {
                display: flex;
                list-style: none;
                gap: 1.5rem;
            }

            .navbar a {
                color: #fff;
                text-decoration: none;
                font-size: 1rem;
                transition: color 0.3s ease;
            }

            .navbar a:hover {
                color: #00d4ff;
            }

            /* Main Content */
            .container {
                max-width: 1200px;
                margin: 100px auto 2rem;
                padding: 0 2rem;
                flex: 1;
            }

            /* Menu Section */
            .menu {
                margin-bottom: 3rem;
            }

            .menu h2 {
                font-size: 2rem;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
            }

            .split-container {
                display: flex;
                gap: 2rem;
                flex-wrap: wrap;
                justify-content: center;
            }

            .video-frame {
                flex: 1;
                min-width: 300px;
                max-width: 600px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transform: translateZ(0);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-align: center;
            }

            .video-frame:hover {
                transform: translateZ(20px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            }

            .video-frame img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }

            .video-frame p {
                font-size: 1.1rem;
                margin-top: 1rem;
                color: #ccc;
            }

            .frame-box {
                flex: 1;
                min-width: 300px;
                max-width: 400px;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transform: translateZ(0);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1.5rem;
            }

            .frame-box:hover {
                transform: translateZ(20px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            }

            .feature-button {
                background: linear-gradient(135deg, #2a5298, #1e3c72);
                border: none;
                color: #fff;
                padding: 1rem 2rem;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1.2rem;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                transition: background 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
                width: 100%;
                max-width: 300px;
                justify-content: center;
                text-align: center;
                text-decoration: none;
            }

            .feature-button:hover {
                background: linear-gradient(135deg, #1e3c72, #00d4ff);
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            }

            .feature-button input[type="file"] {
                display: none;
            }

            .feature-button label {
                cursor: pointer;
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
            }

            .menu-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transform: translateZ(0);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-align: center;
                max-width: 300px;
                margin: 1rem auto;
            }

            .menu-card:hover {
                transform: translateZ(20px) scale(1.05);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            }

            .menu-card a {
                color: #fff;
                text-decoration: none;
                font-size: 1.2rem;
                display: block;
            }

            /* About Section */
            .about {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 3rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transform: translateZ(0);
                transition: transform 0.3s ease;
            }

            .about:hover {
                transform: translateZ(20px);
            }

            .about h2 {
                font-size: 2rem;
                margin-bottom: 1rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
            }

            .about p {
                font-size: 1.1rem;
                line-height: 1.6;
            }

            /* Features Section */
            .features {
                margin-bottom: 3rem;
            }

            .features h2 {
                font-size: 2rem;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
            }

            .feature-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transform: translateZ(0);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-align: center;
            }

            .feature-card:hover {
                transform: translateZ(20px) scale(1.05);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            }

            .feature-card h3 {
                font-size: 1.3rem;
                margin-bottom: 0.5rem;
            }

            .feature-card p {
                font-size: 1rem;
            }

            /* Team Section */
            .team {
                margin-bottom: 3rem;
            }

            .team h2 {
                font-size: 2rem;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
            }

            .team-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
            }

            .team-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transform: translateZ(0);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-align: center;
                perspective: 1000px;
            }

            .team-card:hover {
                transform: translateZ(20px) rotateY(10deg) scale(1.05);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            }

            .team-card img {
                width: 150px;
                height: 150px;
                border-radius: 50%;
                object-fit: cover;
                margin-bottom: 1rem;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease;
            }

            .team-card img:hover {
                transform: scale(1.1);
            }

            .team-card h3 {
                font-size: 1.3rem;
                margin-bottom: 0.5rem;
            }

            .team-card p {
                font-size: 1rem;
                color: #ccc;
            }

            /* Footer */
            .footer {
                background: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(10px);
                padding: 2rem;
                text-align: center;
                box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.3);
            }

            .footer p {
                font-size: 1rem;
                margin-bottom: 0.5rem;
            }

            .footer a {
                color: #00d4ff;
                text-decoration: none;
                margin: 0 0.5rem;
            }

            .footer a:hover {
                text-decoration: underline;
            }

            /* Responsive Design */
            @media (max-width: 900px) {
                .navbar {
                    flex-direction: column;
                    gap: 1rem;
                }

                .navbar ul {
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .container {
                    margin-top: 120px;
                }

                .split-container {
                    flex-direction: column;
                    align-items: center;
                }

                .video-frame, .frame-box {
                    max-width: 100%;
                }
            }

            @media (max-width: 600px) {
                .navbar h1 {
                    font-size: 1.5rem;
                }

                .navbar a {
                    font-size: 0.9rem;
                }

                .menu h2, .about h2, .features h2, .team h2 {
                    font-size: 1.8rem;
                }

                .about p {
                    font-size: 1rem;
                }

                .feature-card h3, .team-card h3 {
                    font-size: 1.2rem;
                }

                .feature-card p, .team-card p {
                    font-size: 0.9rem;
                }

                .feature-button {
                    font-size: 1rem;
                    padding: 0.8rem 1.5rem;
                }

                .menu-card a {
                    font-size: 1rem;
                }

                .video-frame p {
                    font-size: 1rem;
                }

                .team-card img {
                    width: 120px;
                    height: 120px;
                }
            }

            @media (max-width: 400px) {
                .navbar h1 {
                    font-size: 1.2rem;
                }

                .navbar a {
                    font-size: 0.8rem;
                }

                .menu h2, .about h2, .features h2, .team h2 {
                    font-size: 1.5rem;
                }

                .about p {
                    font-size: 0.9rem;
                }

                .feature-card h3, .team-card h3 {
                    font-size: 1rem;
                }

                .feature-card p, .team-card p {
                    font-size: 0.8rem;
                }

                .feature-button {
                    font-size: 0.9rem;
                    padding: 0.7rem 1.2rem;
                }

                .menu-card a {
                    font-size: 0.9rem;
                }

                .video-frame p {
                    font-size: 0.9rem;
                }

                .team-card img {
                    width: 100px;
                    height: 100px;
                }

                .footer p {
                    font-size: 0.9rem;
                }
            }
        </style>
        <script>
            // JavaScript to handle dynamic image updates
            function updateMediaDisplay(src) {
                const mediaElement = document.getElementById('mediaDisplay');
                mediaElement.src = src;
                mediaElement.style.display = 'block';
            }

            // Function to reset media display
            function resetMediaDisplay() {
                const mediaElement = document.getElementById('mediaDisplay');
                mediaElement.src = '';
                mediaElement.style.display = 'none';
                document.getElementById('mediaCaption').textContent = 'Select an option to view media';
            }

            // Run on page load
            document.addEventListener('DOMContentLoaded', function() {
                resetMediaDisplay();
            });
        </script>
    </head>
    <body>
        <!-- Navbar -->
        <nav class="navbar">
            <h1>Vision to Voice</h1>
            <ul>
                <li><a href="#menu">Try It </a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#features">Features</a></li>
                <li><a href="#team">Team</a></li>
            </ul>
        </nav>

        <!-- Main Content -->
        <div class="container">
            <!-- Menu Section -->
            <section class="menu" id="menu">
                <h2>Try It Out</h2>
                <div class="split-container">
                    <div class="video-frame">
                        <img id="mediaDisplay" src="" alt="Media Display" style="display: none;">
                        <p id="mediaCaption">Select an option to view media</p>
                    </div>
                    <div class="frame-box">
                        <a href="{{ url_for('webcam') }}" class="feature-button" onclick="updateMediaDisplay('{{ url_for('stream') }}')">🎙️ Use Live Webcam</a>
                        <form action="{{ url_for('upload_video') }}" method="post" enctype="multipart/form-data">
                            <input type="file" name="video_file" id="video_file" accept="video/*" required style="display: none;" onchange="this.form.submit(); updateMediaDisplay('{{ url_for('stream') }}')">
                            <button type="submit" class="feature-button">
                                <label for="video_file">🎬 Upload Video</label>
                            </button>
                        </form>
                        <form action="{{ url_for('upload_image') }}" method="post" enctype="multipart/form-data">
                            <input type="file" name="image_file" id="image_file" accept="image/*" required style="display: none;" onchange="this.form.submit()">
                            <button type="submit" class="feature-button">
                                <label for="image_file">🖼️ Upload Image</label>
                            </button>
                        </form>
                        <div class="menu-card">
                            <a href="{{ url_for('stop_streaming') }}" onclick="resetMediaDisplay()">🚪 Stop Streaming</a>
                        </div>
                    </div>
                </div>
            </section>

            <!-- About Section -->
            <section class="about" id="about">
                <h2>About the Project</h2>
                <p>
                    Vision to Voice is an innovative AI-powered application that transforms visual content into spoken narratives in real-time. 
                    Using advanced computer vision and natural language processing, it generates descriptive captions for images and videos, 
                    which are then converted to speech. Whether you're using a live webcam feed, uploading a video, or processing a single image, 
                    this tool provides an accessible way to understand visual scenes through audio descriptions.
                </p>
                <p>
                    Built with the BLIP model for image captioning, Sentence-Transformers for semantic similarity, OpenCV for image/video processing, 
                    and pyttsx3 for text-to-speech, Vision to Voice is designed to be both powerful and user-friendly.
                </p>
            </section>

            <!-- Features Section -->
            <section class="features" id="features">
                <h2>Features</h2>
                <div class="feature-grid">
                    <div class="feature-card">
                        <h3>Real-Time Captioning</h3>
                        <p>Generates descriptive captions for live webcam feeds and videos as scenes change.</p>
                    </div>
                    <div class="feature-card">
                        <h3>Text-to-Speech</h3>
                        <p>Converts captions into spoken audio for an accessible experience.</p>
                    </div>
                    <div class="feature-card">
                        <h3>Multi-Source Support</h3>
                        <p>Processes live webcam feeds, video files, and static images seamlessly.</p>
                    </div>
                    <div class="feature-card">
                        <h3>Intelligent Filtering</h3>
                        <p>Uses semantic similarity to avoid repetitive captions, ensuring meaningful narration.</p>
                    </div>
                </div>
            </section>

            <!-- Team Section -->
            <section class="team" id="team">
                <h2>Our Team</h2>
                <div class="team-grid">
                    <div class="team-card">
                        <img src="{{ url_for('static', filename='img/1.jpg') }}" alt="Pushkar Singh">
                        <h3>Pushkar Singh</h3>
                        <p>Lead Developer with expertise in computer vision and Flask. Passionate about creating accessible AI solutions.</p>
                    </div>
                    <div class="team-card">
              <img src="{{ url_for('static', filename='img/mypicture.jpg') }}" alt="Vishal Sharma">

                        <h3>vishal Sharma</h3>
                        <p>AI Specialist focused on natural language processing and model optimization. Loves building intuitive interfaces.</p>
                    </div>
                    <div class="team-card">
                        <img src="{{ url_for('static', filename='img/A.jpg') }}" alt="Aashutosh Kumar">
                        <h3>Aashutosh Kumar</h3>
                        <p>Frontend Designer with a knack for creating visually appealing and user-friendly web experiences.</p>
                    </div>
                </div>
            </section>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <p>Developed by <a href="https://github.com/El-lunatico">El-lunatico</a> | Vision to Voice &copy; 2025</p>
            <p>
                <a href="https://github.com/El-lunatico/vision-to-voice">GitHub</a> |
                <a href="mailto:contact@visiontovoice.com">Contact</a> |
                <a href="#about">About</a> |
                <a href="#team">Team</a>
            </p>
        </footer>
    </body>
    </html>
    """)

@app.route('/webcam')
def webcam():
    """Starts webcam processing in a separate thread."""
    global stop_stream, output_queue
    stop_stream = False
    output_queue = Queue()  # Reset queue
    threading.Thread(target=process_video_flask, args=(WEBCAM_INDEX, models), daemon=True).start()
    return redirect(url_for('index'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handles video file upload and starts processing in a separate thread."""
    global stop_stream, output_queue
    if 'video_file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['video_file']
    if file.filename == '':
        return redirect(url_for('index'))
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    stop_stream = False
    output_queue = Queue()  # Reset queue
    threading.Thread(target=process_video_flask, args=(file_path, models), daemon=True).start()
    return redirect(url_for('index'))

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handles image file upload and processes it."""
    global output_queue
    if 'image_file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image_file']
    if file.filename == '':
        return redirect(url_for('index'))
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    output_queue = Queue()  # Reset queue
    threading.Thread(target=process_image_flask, args=(file_path, models[1], models[2], models[0]), daemon=True).start()
    
    # Wait briefly for processing to complete
    import time
    time.sleep(2)  # Adjust based on expected processing time
    
    if not output_queue.empty():
        result = output_queue.get()
        if result[0] == "error":
            return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error</title>
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                        font-family: 'Arial', sans-serif;
                    }

                    body {
                        background: url('https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?q=80&w=2070&auto=format&fit=crop') no-repeat center center fixed;
                        background-size: cover;
                        color: #fff;
                        min-height: 100vh;
                        display: flex;
                        flex-direction: column;
                        position: relative;
                    }

                    body::before {
                        content: '';
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.4);
                        z-index: -1;
                    }

                    @supports not (background: url('')) {
                        body {
                            background: linear-gradient(135deg, #1e3c72, #2a5298);
                        }
                    }

                    .navbar {
                        position: fixed;
                        top: 0;
                        width: 100%;
                        background: rgba(0, 0, 0, 0.7);
                        backdrop-filter: blur(10px);
                        padding: 1rem 2rem;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        z-index: 1000;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                    }

                    .navbar h1 {
                        font-size: 1.8rem;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                    }

                    .navbar ul {
                        display: flex;
                        list-style: none;
                        gap: 1.5rem;
                    }

                    .navbar a {
                        color: #fff;
                        text-decoration: none;
                        font-size: 1rem;
                        transition: color 0.3s ease;
                    }

                    .navbar a:hover {
                        color: #00d4ff;
                    }

                    .container {
                        max-width: 1200px;
                        margin: 80px auto 2rem;
                        padding: 0 2rem;
                        flex: 1;
                        text-align: center;
                    }

                    h1 {
                        font-size: 2.5rem;
                        margin-bottom: 1rem;
                        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
                        transform: translateZ(20px);
                    }

                    p {
                        font-size: 1.2rem;
                        margin-bottom: 2rem;
                    }

                    .card {
                        background: rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                        transform: translateZ(0);
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                        max-width: 500px;
                        margin: 0 auto;
                    }

                    .card:hover {
                        transform: translateZ(20px) scale(1.05);
                        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
                    }

                    .card a {
                        color: #fff;
                        text-decoration: none;
                        font-size: 1.2rem;
                        display: block;
                    }

                    .footer {
                        background: rgba(0, 0, 0, 0.7);
                        backdrop-filter: blur(10px);
                        padding: 2rem;
                        text-align: center;
                        box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.3);
                    }

                    .footer p {
                        font-size: 1rem;
                        margin-bottom: 0.5rem;
                    }

                    .footer a {
                        color: #00d4ff;
                        text-decoration: none;
                        margin: 0 0.5rem;
                    }

                    .footer a:hover {
                        text-decoration: underline;
                    }

                    @media (max-width: 900px) {
                        .navbar {
                            flex-direction: column;
                            gap: 1rem;
                        }

                        .navbar ul {
                            flex-wrap: wrap;
                            justify-content: center;
                        }

                        .container {
                            margin-top: 120px;
                        }
                    }

                    @media (max-width: 600px) {
                        .navbar h1 {
                            font-size: 1.5rem;
                        }

                        .navbar a {
                            font-size: 0.9rem;
                        }

                        h1 {
                            font-size: 1.8rem;
                        }

                        p {
                            font-size: 1rem;
                        }

                        .card {
                            padding: 1rem;
                        }

                        .card a {
                            font-size: 1rem;
                        }
                    }

                    @media (max-width: 400px) {
                        .navbar h1 {
                            font-size: 1.2rem;
                        }

                        .navbar a {
                            font-size: 0.8rem;
                        }

                        h1 {
                            font-size: 1.5rem;
                        }

                        p {
                            font-size: 0.9rem;
                        }

                        .card {
                            padding: 0.8rem;
                        }

                        .card a {
                            font-size: 0.9rem;
                        }

                        .footer p {
                            font-size: 0.9rem;
                        }
                    }
                </style>
            </head>
            <body>
                <nav class="navbar">
                    <h1>Vision to Voice</h1>
                    <ul>
                        <li><a href="{{ url_for('index') }}#menu">Try It Out</a></li>
                        <li><a href="{{ url_for('index') }}#about">About</a></li>
                        <li><a href="{{ url_for('index') }}#features">Features</a></li>
                        <li><a href="{{ url_for('index') }}#team">Team</a></li>
                    </ul>
                </nav>

                <div class="container">
                    <h1>Error</h1>
                    <p>{{ caption }}</p>
                    <div class="card">
                        <a href="{{ url_for('index') }}">Back to Menu</a>
                    </div>
                </div>

                <footer class="footer">
                    <p>Developed by <a href="https://github.com/El-lunatico">El-lunatico</a> | Vision to Voice &copy; 2025</p>
                    <p>
                        <a href="https://github.com/El-lunatico/vision-to-voice">GitHub</a> |
                        <a href="mailto:contact@visiontovoice.com">Contact</a> |
                        <a href="{{ url_for('index') }}#about">About</a> |
                        <a href="{{ url_for('index') }}#team">Team</a>
                    </p>
                </footer>
            </body>
            </html>
            """, caption=result[1])
        elif result[0] == "image":
            _, caption, output_path = result
            with open(output_path, 'rb') as f:
                img_data = f.read()
            img_str = b64encode(img_data).decode('utf-8')
            return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Processed Image</title>
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                        font-family: 'Arial', sans-serif;
                    }

                    body {
                        background: url('https://images.unsplash.com/photo-1515694346937-94d85e41e6f0?q=80&w=2070&auto=format&fit=crop') no-repeat center center fixed;
                        background-size: cover;
                        color: #fff;
                        min-height: 100vh;
                        display: flex;
                        flex-direction: column;
                        position: relative;
                    }

                    body::before {
                        content: '';
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.4);
                        z-index: -1;
                    }

                    @supports not (background: url('')) {
                        body {
                            background: linear-gradient(135deg, #1e3c72, #2a5298);
                        }
                    }

                    .navbar {
                        position: fixed;
                        top: 0;
                        width: 100%;
                        background: rgba(0, 0, 0, 0.7);
                        backdrop-filter: blur(10px);
                        padding: 1rem 2rem;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        z-index: 1000;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                    }

                    .navbar h1 {
                        font-size: 1.8rem;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                    }

                    .navbar ul {
                        display: flex;
                        list-style: none;
                        gap: 1.5rem;
                    }

                    .navbar a {
                        color: #fff;
                        text-decoration: none;
                        font-size: 1rem;
                        transition: color 0.3s ease;
                    }

                    .navbar a:hover {
                        color: #00d4ff;
                    }

                    .container {
                        max-width: 1200px;
                        margin: 80px auto 2rem;
                        padding: 0 2rem;
                        flex: 1;
                        text-align: center;
                    }

                    h1 {
                        font-size: 2.5rem;
                        margin-bottom: 1rem;
                        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
                        transform: translateZ(20px);
                    }

                    p {
                        font-size: 1.2rem;
                        margin-bottom: 1.5rem;
                        transform: translateZ(10px);
                    }

                    img {
                        max-width: 100%;
                        height: auto;
                        border-radius: 15px;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                        transform: translateZ(0);
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                    }

                    img:hover {
                        transform: translateZ(20px) scale(1.02);
                        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                    }

                    .card {
                        background: rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                        transform: translateZ(0);
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                        margin-top: 2rem;
                        max-width: 500px;
                        margin-left: auto;
                        margin-right: auto;
                    }

                    .card:hover {
                        transform: translateZ(20px) scale(1.05);
                        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
                    }

                    .card a {
                        color: #fff;
                        text-decoration: none;
                        font-size: 1.2rem;
                        display: block;
                    }

                    .footer {
                        background: rgba(0, 0, 0, 0.7);
                        backdrop-filter: blur(10px);
                        padding: 2rem;
                        text-align: center;
                        box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.3);
                    }

                    .footer p {
                        font-size: 1rem;
                        margin-bottom: 0.5rem;
                    }

                    .footer a {
                        color: #00d4ff;
                        text-decoration: none;
                        margin: 0 0.5rem;
                    }

                    .footer a:hover {
                        text-decoration: underline;
                    }

                    @media (max-width: 900px) {
                        .navbar {
                            flex-direction: column;
                            gap: 1rem;
                        }

                        .navbar ul {
                            flex-wrap: wrap;
                            justify-content: center;
                        }

                        .container {
                            margin-top: 120px;
                        }
                    }

                    @media (max-width: 600px) {
                        .navbar h1 {
                            font-size: 1.5rem;
                        }

                        .navbar a {
                            font-size: 0.9rem;
                        }

                        h1 {
                            font-size: 1.8rem;
                        }

                        p {
                            font-size: 1rem;
                        }

                        img {
                            max-width: 90%;
                        }

                        .card {
                            padding: 1rem;
                        }

                        .card a {
                            font-size: 1rem;
                        }
                    }

                    @media (max-width: 400px) {
                        .navbar h1 {
                            font-size: 1.2rem;
                        }

                        .navbar a {
                            font-size: 0.8rem;
                        }

                        h1 {
                            font-size: 1.5rem;
                        }

                        p {
                            font-size: 0.9rem;
                        }

                        img {
                            max-width: 85%;
                        }

                        .card {
                            padding: 0.8rem;
                        }

                        .card a {
                            font-size: 0.9rem;
                        }

                        .footer p {
                            font-size: 0.9rem;
                        }
                    }
                </style>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        const imgSrc = "{{ img_str | default('') }}";
                        if (imgSrc) {
                            const mediaElement = document.getElementById('mediaDisplay');
                            mediaElement.src = "data:image/jpeg;base64,{{ img_str }}";
                            mediaElement.style.display = 'block';
                            document.getElementById('mediaCaption').textContent = "Caption: {{ caption }}";
                        }
                    });
                </script>
            </head>
            <body>
                <nav class="navbar">
                    <h1>Vision to Voice</h1>
                    <ul>
                        <li><a href="{{ url_for('index') }}#menu">Try It </a></li>
                        <li><a href="{{ url_for('index') }}#about">About</a></li>
                        <li><a href="{{ url_for('index') }}#features">Features</a></li>
                        <li><a href="{{ url_for('index') }}#team">Team</a></li>
                    </ul>
                </nav>

                <div class="container">
                    <h1>Processed Image</h1>
                    <div class="video-frame">
                        <img id="mediaDisplay" src="data:image/jpeg;base64,{{ img_str }}" alt="Processed Image">
                        <p id="mediaCaption">Caption: {{ caption }}</p>
                    </div>
                    <div class="card">
                        <a href="{{ url_for('index') }}">Back to Menu</a>
                    </div>
                </div>

                <footer class="footer">
                    <p>Developed by <a href="https://github.com/El-lunatico">vision to voice team </a> | Vision to Voice &copy; 2025</p>
                    <p>
                        <a href="https://github.com/El-lunatico/vision-to-voice">GitHub</a> |
                        <a href="mailto:contact@visiontovoice.com">Contact</a> |
                        <a href="{{ url_for('index') }}#about">About</a> |
                        <a href="{{ url_for('index') }}#team">Team</a>
                    </p>
                </footer>
            </body>
            </html>
            """, caption=caption, img_str=img_str)
    return redirect(url_for('index'))

@app.route('/stream')
def stream():
    """Streams video frames from the output queue."""
    def generate():
        while not stop_stream:
            if not output_queue.empty():
                result = output_queue.get()
                if result[0] == "frame":
                    _, _, output_path = result
                    with open(output_path, 'rb') as f:
                        frame_bytes = f.read()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                elif result[0] == "end":
                    break
            else:
                import time
                time.sleep(0.01)  # Prevent busy looping
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + b"Stream ended" + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_streaming')
def stop_streaming():
    """Stops the video streaming."""
    global stop_stream
    stop_stream = True
    return redirect(url_for('index'))

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # Load models once at startup
    models = setup_models()
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)