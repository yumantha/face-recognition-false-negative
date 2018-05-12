import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
path = 'dataset'


def getImagesByID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    ids = []
    # names = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        # name = str(os.path.split(imagePath)[-1].split('.')[2])

        faces.append(faceNp)
        ids.append(id)
        # names.append(name)

        cv2.imshow("training", faceNp)
        cv2.waitKey(10)

    # return np.array(ids), faces, names
    return np.array(ids), faces


# ids, faces, names = getImagesByID(path)
ids, faces = getImagesByID(path)

recognizer.train(faces, ids)
recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()
