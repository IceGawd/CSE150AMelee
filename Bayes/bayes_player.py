import melee
import signal
from dataset_collector import *
import numpy as np
import random
import pickle
import sys
import time
import math
from sklearn.cluster import MiniBatchKMeans

maxVal = None
minVal = None

def normalize(point):
	global minVal, maxVal
	point = (np.array(point) - minVal) / (maxVal - minVal)
	return np.nan_to_num(point, nan=0.5, posinf=1, neginf=0)

def data_normalization(data):
	global maxVal, minVal

	print("Data Normalization!")

	p = 0
	keys = data["data"].keys()
	for i, key in enumerate(keys):
		while (50 * i > p * len(keys)):
			print(str(p) + "%")
			p += 1

		lists = [d["value"] for d in data["data"][key]]

		if (maxVal == None):
			maxVal = list(map(max, zip(*lists)))
			minVal = list(map(min, zip(*lists)))
		else:
			maxVal = list(map(max, zip(maxVal, *lists)))
			minVal = list(map(min, zip(minVal, *lists)))

	maxVal = np.array(maxVal)
	minVal = np.array(minVal)

	for i, key in enumerate(keys):
		while (50 * i > (p - 50) * len(keys)):
			print(str(p) + "%")
			p += 1

		# print(key)
		for d in data["data"][key]:
			d["value"] = normalize(d["value"])

def bad_compression(data):
	for key in data["data"].keys():
		data["data"][key] = data["data"][key][-1000:]

	return data


def fast_compression(data):
	p = 0
	keys = data["data"].keys()
	for i, key in enumerate(keys):
		while (100 * i > p * len(keys)):
			print(str(p) + "%")
			p += 1

		ddk = data["data"][key]
		differences = []

		for index in range(0, len(ddk) - 1):
			differences.append(np.linalg.norm(getArray(ddk[index + 1]) - getArray(ddk[index])))

		threshold = np.mean(differences)

		newPoints = []
		start = 0
		for index in range(0, len(ddk) - 1):
			if threshold < differences[index]:
				newPoints.append(pointAverager(ddk[start:index + 1]))
				start = index + 1

		newPoints.append(pointAverager(ddk[start:len(ddk)]))

		# print("Compression Rate: " + str(len(newPoints) / len(ddk)))
		data["data"][key] = newPoints

	return data


savefile = "ice_god_falco"
filename = "./pickles/compressed_" + savefile + ".pkl"
connect_code = "QHAS#352"
# connect_code = ""

character_stage = {
	"ice_god_samus": [melee.Character.SAMUS, melee.Stage.RANDOM_STAGE], 
	"samus_marth": [melee.Character.SAMUS, melee.Stage.RANDOM_STAGE], 
	"fox_fox_FD": [melee.Character.FOX, melee.Stage.FINAL_DESTINATION], 
	"ice_god_fox": [melee.Character.FOX, melee.Stage.RANDOM_STAGE], 
	"ice_god_falco": [melee.Character.FALCO, melee.Stage.RANDOM_STAGE],
	"falco_falcon": [melee.Character.FALCO, melee.Stage.RANDOM_STAGE],
	"falco_marth": [melee.Character.FALCO, melee.Stage.RANDOM_STAGE],
	"fox_falco": [melee.Character.FOX, melee.Stage.RANDOM_STAGE], 
	"testing": [melee.Character.MARTH, melee.Stage.FINAL_DESTINATION], 
}

character, stage = character_stage[savefile]
# costume = random.randint(0, 4)
costume = 1

if os.path.exists(filename):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		maxVal = data["maxVal"]
		minVal = data["minVal"]
else:
	data = loadData(savefile)
	data_normalization(data)
	fast_compression(data)

	data["minVal"] = minVal
	data["maxVal"] = maxVal

	with open(filename, 'wb') as f:
		pickle.dump(data, f)


def weighted_random(values, weights):
	r = random.random()

	i = -1
	while r > 0:
		i += 1
		r -= weights[i]

	return values[i]

def set_controller_state(controller, cs):
	controller.release_all()

	controller.tilt_analog(melee.Button.BUTTON_MAIN, cs.main_stick[0], cs.main_stick[1])
	controller.tilt_analog(melee.Button.BUTTON_C, cs.c_stick[0], cs.c_stick[1])
	for b in cs.button:
		if (b != melee.Button.BUTTON_START and b != melee.Button.BUTTON_L) and b != melee.Button.BUTTON_R:
			if cs.button[b]:
				controller.press_button(b)
			else:
				controller.release_button(b)

	# controller.press_shoulder(melee.Button.BUTTON_L, cs.l_shoulder)
	# controller.press_shoulder(melee.Button.BUTTON_R, cs.r_shoulder)

	controller.flush()

console = melee.Console(path="/home/avighna/Downloads/Slippi_Online-x86_64.AppImage")

myPort = 1
opPort = 2

controller = melee.Controller(console=console, port=myPort)
controller_human = melee.Controller(console=console,
									port=opPort,
									type=melee.ControllerType.GCN_ADAPTER)

console.run()
console.connect()

controller.connect()
controller_human.connect()

def signal_handler(sig, frame):
	console.stop()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
	# start = time.time()

	gamestate = console.step()
	if gamestate is None:
		continue

	if console.processingtime * 1000 > 17:
		print("WARNING: Last frame took " + str(console.processingtime*1000) + "ms to process.")

	if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
		if connect_code != "":
			discovered_port = melee.gamestate.port_detector(gamestate, character, costume)
			if (discovered_port != myPort):
				opPort = myPort
				myPort = discovered_port

		key = stringEnumerate(keyInfo(gamestate, myPort, opPort))

		print(key)

		if (key in data["data"]):
			ddk = data["data"][key]
			# print(len(ddk))

			vDict = valueFn(gamestate, myPort, opPort)
			# print("preval: " + str(vDict["value"]))
			vDict["value"] = normalize(vDict["value"])
			# print("postval: " + str(vDict["value"]))

			total = 0
			weights = []
			seenIndicies = []
			for i, value in enumerate(ddk):
				w = 1 / np.linalg.norm((vDict["value"] - value["value"]) * valueWeighting) 

				# if (gamestate.players[myPort].on_ground and value["input"].button[melee.Button.BUTTON_R]) or (gamestate.players[myPort].on_ground and value["input"].button[melee.Button.BUTTON_X]):
				# 	seenIndicies.append(i)
					# print("value: " + str(value["value"]))
					# print("valueWeighting: " + str(valueWeighting))
					# print("norm: " + str(np.linalg.norm((vDict["value"] - value["value"]) * valueWeighting)))
					# print("Weight: " + str(w))

				weights.append(w)
				total += w

			probabilities = np.array(weights) / total
			# if (len(seenIndicies) > 0):
			# 	print("Probability: " + str(probabilities[seenIndicies[0]]))

			choice = weighted_random(ddk, probabilities.tolist())
			set_controller_state(controller, choice["input"])

		"""
		if gamestate.players[myPort].off_stage:
			controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 1)
			controller.press_button(melee.Button.BUTTON_B)
		"""

	else:
		melee.MenuHelper.menu_helper_simple(gamestate,
											controller,
											character,
											stage,
											connect_code,
											costume=costume,
											autostart=True,
											swag=False)

	# end = time.time()

	# print("Frame: " + str(end - start))