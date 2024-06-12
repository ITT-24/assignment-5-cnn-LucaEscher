[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/GaaycKto)


# Task 1

See hyperparameters.ipynb.

# Task 2

## Take pictures

To take pictures call take_picture.py like the following:

```
 python3 02-dataset/take_picture.py ./02-dataset/data-escher/peace peace3.jpg
```

### Generall description:
```
python script_name.py /path/to/output_folder image_name.jpg
```

Replace script_name.py with the name of your script file, /path/to/output_folder with the desired output folder, and image_name.jpg with the desired file name for the captured image. If you omit the filename, it will default to captured_image.jpg.

## Annotate

Call annotate.py to create annotations for one folder. You can specify the folder you want to pick inside of the script with the parameter CONDITION.

Expected are directories which have pictures stored. The pircutres are expected to be named with the foldername they are in and a number. For example in the folder stop are pictures like stop1, stop2, stop3 and so on.

**A JSON File for each annotation will be created. I then fused all condition annotations together to one file called annot-escher.json**

## Output

You can find the final confusion matrix at 02-dataset/conf-matrix.png.

You can also find the trained model at 02-dataset/my_model.keras.


# Task 3

**Tipp: Hold a big white paper behind your hand which makes the gesture! This way the recognition is way better!**

You can either train and use a new model or use the already created model.

Call the function like this:
```
python3 media_control.py False ../data/gesture_dataset_sample 1
```

## Params:

```
python3 media_control.py <model_ready> <data_path> <video_id>
```

#### model_ready:

- Type: Boolean 
- Indicates whether to load a pre-trained model or to train a new one.

#### Values:
True: Load a pre-trained model.
False: Train a new model from the dataset.


#### data_path:

- Type: String
- The path to the dataset HaGRID.
- Example: ../data/gesture_dataset_sample
 
#### video_id:

- Type: Integer
- The ID of the video device to use for capturing input gestures. Typically, 0 refers to the default webcam.
- Example: 0 for the default webcam, 1 for an external webcam.
