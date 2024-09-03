from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import face_recognition
from PIL import Image
import io
from mtcnn import MTCNN
import cv2
import faiss
import os
import imgaug.augmenters as iaa

app = FastAPI()

# Load encodings from the file
def load_encodings(file_path):
    if not os.path.exists(file_path):
        return np.array([]), []
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return np.array(data["encodings"]), data["labels"]

# Save encodings to the file
def save_encodings(encodings, labels, file_path):
    data = {"encodings": encodings, "labels": labels}
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

# Detect and align face
def detect_and_align_face(image):
    detector = MTCNN()  # Initialize the MTCNN face detector
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  # Convert the image to RGB format
    detections = detector.detect_faces(image_rgb)  # Detect faces in the image

    if len(detections) == 0:
        raise ValueError("No face detected in the image.")

    detection = detections[0]  # Assume the first detected face
    x, y, width, height = detection['box']  # Get the bounding box of the face
    keypoints = detection['keypoints']  # Get facial keypoints (eyes, nose, mouth)
    face = image_rgb[y:y + height, x:x + width]  # Extract the face from the image

    # Calculate the angle to align the face based on eye positions
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.arctan2(delta_y, delta_x) * (180.0 / np.pi)

    # Compute the center of the face and create a rotation matrix
    center = ((x + x + width) // 2, (y + y + height) // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Rotate the image to align the face
    aligned_image = cv2.warpAffine(image_rgb, rot_matrix, (image_rgb.shape[1], image_rgb.shape[0]))
    aligned_face = aligned_image[y:y + height, x:x + width]  # Extract the aligned face

    return Image.fromarray(aligned_face)  # Convert to PIL Image format and return

# Create FAISS index
def create_faiss_index(known_encodings):
    dimension = known_encodings.shape[1]  # Get the dimensionality of the encodings
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index using L2 distance
    index.add(known_encodings)  # Add known encodings to the index
    return index  # Return the FAISS index

# Augment image function
def augment_image(image, num_augmented=5):
    """
    Apply data augmentation to an image.
    
    Parameters:
    image (PIL.Image): The image to augment.
    num_augmented (int): Number of augmented images to generate.
    
    Returns:
    List[PIL.Image]: List of augmented images.
    """
    image = np.array(image)

    # Define a sequence of augmentation techniques
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-25, 25)),  # rotation
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # noise
        iaa.Multiply((0.8, 1.2)),  # brightness
        iaa.GaussianBlur(sigma=(0.0, 1.0))  # blur
    ])

    # Generate augmented images
    augmented_images = [Image.fromarray(aug(image=image)) for _ in range(num_augmented)]
    return augmented_images

# Endpoint to process and save augmented encodings
@app.post("/create/")
async def preprocess_and_save_augmented_encodings(image: UploadFile = File(...), num_augmented: int = 5):
    known_encodings = []
    known_labels = []

    # Load the uploaded image
    image_bytes = await image.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure the image is in RGB format

    # Augment the image
    augmented_images = augment_image(original_image, num_augmented=num_augmented)

    # Include the original image in the list of images to encode
    images_to_encode = [original_image] + augmented_images

    for img in images_to_encode:
        img_array = np.array(img)
        # Encode the face
        encodings = face_recognition.face_encodings(img_array)
        if encodings:
            encoding = encodings[0]
            # Store the encoding and the corresponding label
            known_encodings.append(encoding)
            known_labels.append(image.filename)  # Use the uploaded image filename as the label

    # Save encodings and labels to a file
    encodings_file = "face_encoding.pkl"
    save_encodings(np.array(known_encodings), known_labels, encodings_file)

    return JSONResponse(content={"status": "Success", "message": "Augmented encodings created and saved."})

@app.post("/encode/")
async def encode_face(image: UploadFile = File(...)):
    # Load the image from the uploaded file
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Align the face
    try:
        aligned_face = detect_and_align_face(pil_image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Load existing encodings
    encodings_file = "face_encoding.pkl"
    known_encodings, known_labels = load_encodings(encodings_file)
    
    # Encode the face
    encodings = face_recognition.face_encodings(np.array(aligned_face))
    if not encodings:
        raise HTTPException(status_code=400, detail="No face encoding found.")
    
    # Append the new encoding and label
    known_encodings = list(known_encodings)
    known_encodings.append(encodings[0])
    known_labels.append(image.filename)
    
    # Save the updated encodings
    save_encodings(np.array(known_encodings), known_labels, encodings_file)
    
    return JSONResponse(content={"status": "Success", "message": "Face encoded and saved."})

@app.post("/match/")
async def match_face(image: UploadFile = File(...), similarity_threshold: float = 70.0):
    # Load the image from the uploaded file
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Align the face
    try:
        aligned_face = detect_and_align_face(pil_image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Load existing encodings
    encodings_file = "face_encoding.pkl"
    known_encodings, known_labels = load_encodings(encodings_file)
    
    if len(known_encodings) == 0:
        raise HTTPException(status_code=400, detail="No known faces in the database. Please add some faces first.")
    
    # Encode the face
    target_encodings = face_recognition.face_encodings(np.array(aligned_face))
    if not target_encodings:
        raise HTTPException(status_code=400, detail="No face encoding found.")
    
    target_encoding = target_encodings[0].reshape(1, -1)
    
    # Create FAISS index and search for the best match
    index = create_faiss_index(np.array(known_encodings))
    distances, indices = index.search(target_encoding, 1)
    
    best_match_index = indices[0][0]
    best_similarity_percentage = (1 - distances[0][0]) * 100
    
    if best_similarity_percentage >= similarity_threshold:
        matched_label = known_labels[best_match_index]
        
        # If similarity is greater than 80%, save the new encoding under the same label
        if 82.0 <= best_similarity_percentage < 100.0:
            known_encodings = list(known_encodings)
            known_encodings.append(target_encoding[0])
            known_labels.append(matched_label)
            save_encodings(np.array(known_encodings), known_labels, encodings_file)

        return JSONResponse(content={"status": "Success", "message": f"Match found: {matched_label}, Similarity: {best_similarity_percentage:.2f}%"})
    else:
        return JSONResponse(content={"status": "Failure", "message": f"Not found:, Similarity: {best_similarity_percentage:.2f}%"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
