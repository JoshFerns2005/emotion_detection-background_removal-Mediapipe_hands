# Emotion Detection & Background Manipulation

# Overview

This project is a real-time emotion detection and background manipulation system using Python. It detects facial emotions, overlays corresponding emojis, removes the background, and allows users to insert different backgrounds. Additionally, users can change backgrounds dynamically by showing a specific number of fingers, leveraging hand tracking.

# Features

Emotion Detection with Emoji Overlay: Detects emotions and displays an appropriate emoji on the face.

Background Removal & Replacement: Uses MediaPipe Selfie Segmentation for real-time background removal and allows inserting custom backgrounds.

Hand Gesture Background Control: Uses MediaPipe Hands to detect the number of fingers shown to change backgrounds dynamically.

Graphical User Interface (GUI): Built with Tkinter for ease of use.

# Technologies Used

Python: Core programming language.

Tkinter: GUI framework.

MediaPipe Selfie Segmentation: For background removal.

MediaPipe Hands: For hand gesture recognition.

Custom Emotion Detection Model: Trained for detecting facial emotions in real-time.

OpenCV: Used for video processing and image manipulation.
