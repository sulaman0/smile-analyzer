import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Define detailed lip landmark indices
LIP_LEFT_CORNER = 61  # Left corner of the lips
LIP_RIGHT_CORNER = 291  # Right corner of the lips

UPPER_LIP_TOP = 13  # Top of the upper lip
UPPER_LIP_LEFT = 185  # Upper lip left side
UPPER_LIP_RIGHT = 195  # Upper lip right side

LOWER_LIP_LEFT = 78  # Lower lip left side
LOWER_LIP_RIGHT = 308  # Lower lip right side
LOWER_LIP_BOTTOM = 14  # Bottom of the lower lip


# Function to detect smile and extract features
def detect_smile(image):
    if not os.path.exists(image):
        print("Image file does not exist.")
        return None, None
    else:
        print("Image found, processing...")

    # Load image
    imageObject = cv2.imread(image)
    h, w, _ = imageObject.shape

    gray = cv2.cvtColor(imageObject, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(imageObject, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get lip points
            lip_left = face_landmarks.landmark[LIP_LEFT_CORNER]
            lip_right = face_landmarks.landmark[LIP_RIGHT_CORNER]

            upper_lip_left = face_landmarks.landmark[UPPER_LIP_LEFT]
            upper_lip_right = face_landmarks.landmark[UPPER_LIP_RIGHT]

            lower_lip_left = face_landmarks.landmark[LOWER_LIP_LEFT]
            lower_lip_right = face_landmarks.landmark[LOWER_LIP_RIGHT]

            upper_lip_top = face_landmarks.landmark[UPPER_LIP_TOP]
            lower_lip_bottom = face_landmarks.landmark[LOWER_LIP_BOTTOM]

            # Convert relative coordinates to absolute pixel positions
            lip_width = np.linalg.norm([(lip_right.x - lip_left.x) * w, (lip_right.y - lip_left.y) * h])
            upper_lip_width = np.linalg.norm(
                [(upper_lip_right.x - upper_lip_left.x) * w, (upper_lip_right.y - upper_lip_left.y) * h])
            lower_lip_width = np.linalg.norm(
                [(lower_lip_right.x - lower_lip_left.x) * w, (lower_lip_right.y - lower_lip_left.y) * h])
            upper_lip_height = np.linalg.norm(
                [(upper_lip_top.x - upper_lip_left.x) * w, (upper_lip_top.y - upper_lip_left.y) * h])
            lower_lip_height = np.linalg.norm(
                [(lower_lip_bottom.x - lower_lip_left.x) * w, (lower_lip_bottom.y - lower_lip_left.y) * h])

            # Print lip dimensions
            print(f"Overall Lip Width: {lip_width:.2f}")
            print(f"Upper Lip Width: {upper_lip_width:.2f}")
            print(f"Lower Lip Width: {lower_lip_width:.2f}")
            print(f"Upper Lip Height: {upper_lip_height:.2f}")
            print(f"Lower Lip Height: {lower_lip_height:.2f}")

            # Draw landmarks on the image
            landmark_points = [LIP_LEFT_CORNER, LIP_RIGHT_CORNER, UPPER_LIP_LEFT, UPPER_LIP_RIGHT, LOWER_LIP_LEFT,
                               LOWER_LIP_RIGHT, UPPER_LIP_TOP, LOWER_LIP_BOTTOM]

            for idx in landmark_points:
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                cv2.circle(imageObject, (x, y), 3, (0, 255, 0), -1)  # Draw green circles on landmarks

            # Show the image with lip landmarks
            cv2.imshow("Lip Landmarks", imageObject)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return lip_width, upper_lip_width, lower_lip_width, upper_lip_height, lower_lip_height

    else:
        print("No face detected.")
def calculate_smile_score(lip_width, upper_lip_width, lower_lip_width, upper_lip_height, lower_lip_height):
    # Calculate the average of all dimensions
    average_value = (lip_width + upper_lip_width + lower_lip_width + upper_lip_height + lower_lip_height) / 5

    # Normalize to 0-100 (Adjust the scaling factor if needed)
    smile_score_100 = min(100, max(0, (average_value / 50) * 100))  # Normalize within 0-100

    # Convert to a 1-10 scale
    smile_score_10 = min(10, max(1, smile_score_100 / 10))

    # return round(smile_score_10, 1), round(smile_score_100, 1)
    return round(smile_score_10, 1)
def write_data_to_csv():
    # Define dataset path
    DATASET_PATH = "./dataset/celebA/img_align_celeba"

    # List all image files
    image_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")]

    # Store extracted features
    data = []

    for image_file in image_files:
        image_path = os.path.join(DATASET_PATH, image_file)
        features = detect_smile(image_path)  # Extract lip features

        if features is not None:
            lip_width, upper_lip_width, lower_lip_width, upper_lip_height, lower_lip_height = features
            smile_score = calculate_smile_score(lip_width, upper_lip_width, lower_lip_width, upper_lip_height, lower_lip_height)
            print(features, smile_score, image_file)
            data.append([image_file, lip_width, upper_lip_width, lower_lip_width, upper_lip_height, lower_lip_height, int(smile_score)])

    # Save data to CSV
    df = pd.DataFrame(data, columns=["image", "lip_width", "upper_lip_width", "lower_lip_width", "upper_lip_height",
                                     "lower_lip_height", "smile_score"])
    df.to_csv("smile_data.csv", index=False)

    print("Dataset saved successfully!")

# write_data_to_csv()
detect_smile('./dataset/celebA/img_align_celeba/000001.jpg')
## 🔹 Step 2: Train the AI Model
