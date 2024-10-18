# PlayingSET
Teaching my computer how to play the game of SET using machine learning


# Getting the Data:

[The Original data](https://www.kaggle.com/datasets/kwisatzhaderach/set-cards) was too much for `tensorflow` to process, so flattening the data was the first goal. 

We can handle this issue by specifying a custom directory structure that flattens some of these sub-folders into one level.

Instead of having **TensorFlow** look for many levels, you can categorize images by combining their class labels into a single folder name (e.g., one_blue_diamond_empty/09.png). This way, TensorFlow only looks at one folder per combination.

```
import os
import shutil

base_dir = 'archive-2'
output_dir = 'flattened_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for number in ['one', 'two', 'three']:
    for color in ['blue', 'green', 'red']:
        for shape in ['diamond', 'oval', 'squiggle']:
            for shading in ['empty', 'full', 'partial']:
                # Path to the original images
                source_dir = os.path.join(base_dir, number, color, shape, shading)
                
                # Create a new folder in the flattened directory combining all class labels
                new_dir_name = f"{number}_{color}_{shape}_{shading}"
                new_dir_path = os.path.join(output_dir, new_dir_name)
                os.makedirs(new_dir_path, exist_ok=True)

                # Move images to the new folder
                for img_file in os.listdir(source_dir):
                    img_source_path = os.path.join(source_dir, img_file)
                    img_dest_path = os.path.join(new_dir_path, img_file)
                    shutil.copy(img_source_path, img_dest_path)
```

Then you can delete the original data set and continue on with the code using 'flattened_images'


