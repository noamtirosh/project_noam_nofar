import csv
import os
import numpy as np


def export_landmarks(scelton_points: np.ndarray, action):
    try:
        kay_points = np.array(
            [[point.x, point.y, point.z, point.visibility] for point in scelton_points]).flatten()
        #kay_points = np.insert(kay_points, 0, 1)
        row = list(kay_points)
        row.insert(0,action)
        #kay_points = [action,0,kay_points]
        # kay_points.insert(0,action)
        with open("coords.csv", mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
    except Exception as e:
        pass


def create_first_row():
    # init csv_file
    landmarks = ["class"]
    for val in range(1, 33 + 1):
        landmarks += ['X{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    with open("coords.csv", mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)