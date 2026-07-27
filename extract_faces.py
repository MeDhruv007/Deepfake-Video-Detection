import os
import glob
import cv2
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Settings
BASE_DIR = '/Users/karnav/Desktop/Projects/deepfakedetection/FaceForensics++_C23'
OUT_DIR = '/Users/karnav/Desktop/Projects/deepfakedetection/data'
FRAMES_PER_VIDEO = 5  # Keeping it low to balance dataset size and extraction speed
MARGIN_RATIO = 0.2

# Path specifically on Mac/Linux OpenCV installation
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def process_video(video_path):
    """
    Opens a video, seeks to random frames, detects the face, and saves it.
    Returns the number of successfully extracted faces.
    """
    # Determine output format
    parts = video_path.replace(BASE_DIR, '').strip(os.path.sep).split(os.path.sep)
    is_fake = not ("original_sequences" in parts or "original" in parts)
    
    out_class_dir = os.path.join(OUT_DIR, 'fake' if is_fake else 'real')
    os.makedirs(out_class_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        return 0

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    # Randomly select a few frame indexes to extract
    frame_idxs = random.sample(range(0, total_frames), min(FRAMES_PER_VIDEO, total_frames))
    extracted_count = 0
    
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            continue
            
        # Select largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        margin_x = int(w * MARGIN_RATIO)
        margin_y = int(h * MARGIN_RATIO)
        
        start_x = max(0, x - margin_x)
        start_y = max(0, y - margin_y)
        end_x = min(frame.shape[1], x + w + margin_x)
        end_y = min(frame.shape[0], y + h + margin_y)
        
        cropped_face = frame[start_y:end_y, start_x:end_x]

        # Convert BGR (OpenCV default) → RGB before saving so that
        # tf.image.decode_png reads the correct channel order during training.
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        
        # Save frame
        out_filename = f"{video_name}_frame{idx}.png"
        out_path = os.path.join(out_class_dir, out_filename)
        cv2.imwrite(out_path, cropped_face_rgb)
        extracted_count += 1
        
    cap.release()
    return extracted_count

def main():
    print(f"Starting Face Extraction from: {BASE_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Extracting {FRAMES_PER_VIDEO} frames per video.\n")
    
    all_videos = glob.glob(os.path.join(BASE_DIR, '**', '*.mp4'), recursive=True)
    print(f"Found {len(all_videos)} videos to process.")
    
    # For a dataset of 7000 videos, doing this sequentially takes forever.
    # We will use ProcessPoolExecutor to max out CPU cores.
    total_extracted = 0
    
    # Use max_workers depending on the system; 4 or 8 is safe.
    with ProcessPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
        futures = {executor.submit(process_video, v_path): v_path for v_path in all_videos}
        
        # Progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Faces"):
            try:
                count = future.result()
                total_extracted += count
            except Exception as e:
                pass # skip failing videos
                
    print(f"\nCompleted! Total faces extracted: {total_extracted}")
    print("Dataset is now ready for efficient model training.")

if __name__ == "__main__":
    main()
