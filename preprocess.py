import librosa
import os
import pandas as pd
import json
import math

DATASET_PATH = "D:/Github/Lie-detection/BagOfLies/"
JSON_PATH = "data2.json"


def get_duration(dataset_path, n_fft=2048, hop_length=512):
    dataset = pd.read_csv(os.path.abspath("BagOfLies/Annotations.csv"))

    duration = []

    for index, row in dataset.iterrows():
        abspath = os.path.abspath(dataset_path + row["video"]).replace('video.mp4', 'audio.wav')

        signal, sr = librosa.load(abspath, sr=22050)
        duration.append(math.ceil(librosa.get_duration(signal, sr, n_fft=n_fft,
                                                       hop_length=hop_length)))
        # samples_per_track = 22050 * duration
        # num_mfcc_vectors_per_track = math.ceil(samples_per_track / hop_length)

    frequent_duration = max(set(duration), key=duration.count)
    print(duration)
    print(frequent_duration)
    print(duration.count(9))


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    dataset = pd.read_csv(os.path.abspath("BagOfLies/Annotations.csv"))
    data = {
        "mapping": ['lie', 'truth'],
        "mfcc": [],
        "labels": []
    }

    for index, row in dataset.iterrows():
        abspath = os.path.abspath(dataset_path + row["video"]).replace('video.mp4', 'audio.wav')

        signal, sr = librosa.load(abspath, sr=22050)
        # duration.append(math.ceil(librosa.get_duration(signal, sr)))
        # samples_per_track = 22050 * duration
        # num_mfcc_vectors_per_track = math.ceil(samples_per_track / hop_length)

        mfcc = librosa.feature.mfcc(signal, sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        data["mfcc"].append(mfcc.tolist())
        data["labels"].append(row["truth"])

        print("processing {}".format(abspath))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    # save_mfcc(DATASET_PATH, JSON_PATH)
    get_duration(DATASET_PATH)