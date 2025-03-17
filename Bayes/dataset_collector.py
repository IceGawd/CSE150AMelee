import os
import melee
import pickle
import time
import numpy as np
import math

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

from sklearn.cluster import MiniBatchKMeans

saveKeys = []

csButtonKeys = None
pickles_dir = "./pickles/"

def pointAverager(points):
	avg_value = np.mean([p["value"] for p in points], axis=0)
	avg_input = np.mean([p["input"].to_numpy() for p in points], axis=0)
	return {"value": avg_value, "input": database.PickleableControllerState(np_array=avg_input)}


def data_compression(data, keys, threshold=2000, percent=False):
	print("Data Compression!")

	p = 0
	for i, key in enumerate(keys):
		while (100 * i > p * len(keys)):
			if (percent):
				print(str(p) + "%")

			p += 1

		points = data["data"][key]
		n = len(points)
		new_points = round(n / math.ceil(n / 2000))

		if (n < threshold):
			continue

		# print(n)

		feature_matrix = np.array([
			getArray(point)
			for point in points
		])

		kmeans = MiniBatchKMeans(n_clusters=new_points, batch_size=1024, random_state=42, n_init=10)
		labels = kmeans.fit_predict(feature_matrix)

		clustered_data = []
		for i in range(new_points):
			cluster_points = [points[j] for j in range(n) if labels[j] == i]

			if (len(cluster_points) > 0):
				clustered_data.append(pointAverager(cluster_points))

		data["data"][key] = clustered_data

	return data

def saveData(file, data):
	global saveKeys
	global pickles_dir

	# print(data)
	print("DON'T QUIT Saving...")

	# data_compression(data, saveKeys, threshold=4000)

	for key in saveKeys:
		with open(pickles_dir + file + "_" + key + ".pkl", 'wb') as f:
			pickle.dump(data["data"][key], f)

	saveKeys = []

	with open(pickles_dir + file + ".pkl", 'wb') as f:
		pickle.dump(data["last_file"], f)
	print("Saved!")

def addData(data, gamestate, myPort, opPort):
	global saveKeys

	key = stringEnumerate(keyInfo(gamestate, myPort, opPort))
	vDict = valueFn(gamestate, myPort, opPort)

	if key in data["data"]:
		data["data"][key].append(vDict)
	else:
		data["data"][key] = [vDict]

	if key not in saveKeys:
		saveKeys.append(key)

def loadData(file):
	global pickles_dir

	print("Loading data...")

	data = {"data": {}, "last_file": None}
	
	last_file_path = f"./pickles/{file}.pkl"
	if os.path.exists(last_file_path):
		with open(last_file_path, 'rb') as f:
			data["last_file"] = pickle.load(f)
	
	files = os.listdir(pickles_dir)

	p = 0
	for i, key_file in enumerate(files):
		while (100 * i > p * len(files)):
			print(str(p) + "%")
			p += 1

		if key_file.startswith(file + "_"):
			key = key_file[len(file) + 1:-4]
			with open(os.path.join(pickles_dir, key_file), 'rb') as f:
				data["data"][key] = pickle.load(f)

	print("Data Loaded!")    
	return data

if __name__ == "__main__":
	loopThrough(addData, saveData, loadData, savefile="shunnash_fox")