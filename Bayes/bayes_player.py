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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

savefile = "ice_god_falco"
filename = "./pickles/compressed_" + savefile + ".pkl"
connect_code = "QHAS#352"
# connect_code = ""

character, stage = character_stage[savefile]
# costume = random.randint(0, 4)
costume = 1

maxVal = None
minVal = None

if os.path.exists(filename):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		maxVal = data["maxVal"]
		minVal = data["minVal"]
else:
	data = loadData(savefile)
	minVal, maxVal = data_normalization(data)
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
			vDict["value"] = normalize(vDict["value"], minVal, maxVal)
			# print("postval: " + str(vDict["value"]))

			total = 0
			weights = []
			seenIndicies = []
			for i, value in enumerate(ddk):
				w = 1 / np.linalg.norm((vDict["value"] - value["value"]) * valueWeighting) 

				# if (gamestate.players[myPort].on_ground and value["input"].button[melee.Button.BUTTON_R]) or (gamestate.players[myPort].on_ground and value["input"].button[melee.Button.BUTTON_X]):
					# seenIndicies.append(i)
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