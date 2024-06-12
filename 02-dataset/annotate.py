import cv2
import json
import os

# generatated with chatgpt and adjusted by me 

CONDITION = 'stop'

bboxes = []
labels = []
drawing = False
ix, iy = -1, -1

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bboxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img2 = img.copy()
            cv2.rectangle(img2, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 2)
        bbox = [min(ix, x), min(iy, y), abs(x - ix), abs(y - iy)]
        bboxes.append(bbox)
        # label = input("Enter label for this bounding box: ")
        # labels.append(label)
        labels.append(f'{CONDITION}')

def get_normalized_bbox(bbox, width, height):
    x, y, w, h = bbox
    return [x / width, y / height, w / width, h / height]

# Load images from a directory
image_dir = f'/Users/luca/Projekte/Master/ITT/W6/assignment-5-cnn-LucaEscher/02-dataset/data-escher/{CONDITION}'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
annotations = {}

for image_file in image_files:
    img = cv2.imread(os.path.join(image_dir, image_file))
    img_height, img_width = img.shape[:2]
    bboxes = []
    labels = []

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    normalized_bboxes = [get_normalized_bbox(bbox, img_width, img_height) for bbox in bboxes]
    annotation = {
        "bboxes": normalized_bboxes,
        "labels": labels,
        "leading_conf": 1.0,  # Assuming leading_conf is 1.0 for all
        "leading_hand": "right",  # Assuming leading_hand is "right" for all
        # add your user_id if you wand
        # "user_id": "your_user_id_here"  # Replace with the actual user_id
    }
    
    key = image_file.split('.')[0]
    
    annotations[key] = annotation

# Save annotations to a JSON file
with open('annotations.json', 'w') as f:
    json.dump(annotations, f, indent=4)
