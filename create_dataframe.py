import pandas as pd 
import os 

def create_dataframe(base_path):
    image_path = []
    labels = []
    for emotion in os.listdir(base_path):
        emotion_path = os.path.join(base_path, emotion)
        if os.path.isdir(emotion_path): # Ensure it's a directory
            for image_file in os.listdir(emotion_path):
                image_path.append(os.path.join(emotion_path, image_file))
                labels.append(emotion)

    df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "label"])
    return df

