import cv2
from keras.models import model_from_json
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageSequence
import mediapipe as mp
import os

# Load the emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to overlay emoji on the image
def overlay_emoji(im, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    if emoji.shape[2] == 4:  # Check if emoji has an alpha channel
        alpha_s = emoji[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            im[y:y+h, x:x+w, c] = (alpha_s * emoji[:, :, c] + alpha_l * im[y:y+h, x:x+w, c])
    else:
        print("Emoji does not have an alpha channel")

# Load emoji images
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
emojis = {
    'angry': cv2.imread('emoji/angry.png', -1),
    'disgust': cv2.imread('emoji/disgust.png', -1),
    'fear': cv2.imread('emoji/fear.png', -1),
    'happy': cv2.imread('emoji/happy.png', -1),
    'neutral': cv2.imread('emoji/neutral.png', -1),
    'sad': cv2.imread('emoji/sad.png', -1),
    'surprise': cv2.imread('emoji/surprise.png', -1)
}

# Check if emojis are loaded correctly
for label, emoji in emojis.items():
    if emoji is None:
        print(f"Failed to load emoji for {label}")
    else:
        print(f"Loaded emoji for {label} with shape {emoji.shape}")

# Tkinter GUI
root = tk.Tk()
root.title("Emotion Detector with Emojis and Background Removal")

show_emoji_var = tk.BooleanVar()
remove_background_var = tk.BooleanVar()
show_emoji_var.set(True)
remove_background_var.set(False)

show_emoji_checkbox = tk.Checkbutton(root, text="Show Emoji Emotion", variable=show_emoji_var)
show_emoji_checkbox.pack()

remove_background_checkbox = tk.Checkbutton(root, text="Remove Background", variable=remove_background_var)
remove_background_checkbox.pack()

# Load background images and sort them
bg_folder = 'bg_images'
bg_files = sorted([f for f in os.listdir(bg_folder) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
bg_files.insert(0, 'None')  # Add an option to select no background
selected_bg_var = tk.StringVar(root)
selected_bg_var.set(bg_files[0])  # default value

bg_option_menu = tk.OptionMenu(root, selected_bg_var, *bg_files)
bg_option_menu.pack()

label = tk.Label(root)
label.pack()

webcam = cv2.VideoCapture(0)
bg_index = 0

def count_fingers(hand_landmarks):
    # Count the number of fingers raised
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

def process_hands(frame, rgb_frame):
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            num_fingers = count_fingers(hand_landmarks)
            if 1 <= num_fingers <= 5:
                selected_bg_var.set(bg_files[num_fingers - 1])
                
def process_emotion(frame, gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        image = gray[y:y+h, x:x+w]
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        if show_emoji_var.get():
            emoji = emojis[prediction_label]
            overlay_emoji(frame, emoji, x, y, w, h)


def process_selfie_segmentation(frame, rgb_frame):
    global bg_index
    results = segmentation_model.process(rgb_frame)
    mask = results.segmentation_mask
    threshold = 0.5
    binary_mask = (mask > threshold).astype(np.uint8)
    mask_3d = np.dstack((binary_mask, binary_mask, binary_mask))
    foreground = frame * mask_3d
    if selected_bg_var.get() != 'None':
        bg_path = os.path.join(bg_folder, selected_bg_var.get())
        if selected_bg_var.get().endswith('.gif'):
            gif = Image.open(bg_path)
            bg_images = [cv2.cvtColor(np.array(frame.copy()), cv2.COLOR_RGB2BGR) for frame in ImageSequence.Iterator(gif)]
            bg_resized = cv2.resize(bg_images[bg_index], (frame.shape[1], frame.shape[0]))
            frame = np.where(mask_3d, foreground, bg_resized)
            bg_index = (bg_index + 1) % len(bg_images)
        else:
            bg_image = cv2.imread(bg_path)
            bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
            frame = np.where(mask_3d, foreground, bg_resized)
    else:
        background = np.ones_like(frame, dtype=np.uint8) * 255
        frame = np.where(mask_3d, foreground, background)
    return frame

def update_frame():
    global bg_index
    ret, frame = webcam.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture video")
        return

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    process_hands(frame, rgb_frame)

    # Process the frame with MediaPipe Selfie Segmentation
    if remove_background_var.get():
        frame = process_selfie_segmentation(frame, rgb_frame)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process the frame for emotion detection
    process_emotion(frame, gray)

    # Convert frame back to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

update_frame()
root.mainloop()