import argparse
import cv2
import os
import time

# generated with chatgpt (see chatgpt.md) and adjusted by me

def capture_and_resize_image(output_folder, filename='captured_image.jpg', width=1920, height=1080):
    # wait a second to get into pose for the picture
    time.sleep(1)
    
    # Check if the output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open a connection to the webcam (0 is usually the built-in webcam) but on mac its 1 because 0 is the camera from the phone
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    # Resize the frame to the desired size (1920x1080)
    resized_frame = cv2.resize(frame, (width, height))

    # Build the full path for the output file
    output_path = os.path.join(output_folder, filename)

    # Save the image
    cv2.imwrite(output_path, resized_frame)
    print(f"Image saved to {output_path}")

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture and resize an image from the webcam.")
    parser.add_argument('output_folder', type=str, help='The folder where the image will be saved')
    parser.add_argument('filename', type=str, nargs='?', default='captured_image.jpg', help='The name of the image file (default: captured_image.jpg)')

    args = parser.parse_args()

    capture_and_resize_image(args.output_folder, args.filename)
