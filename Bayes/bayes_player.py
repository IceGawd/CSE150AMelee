import melee
import signal
from dataset_collector import *
import numpy as np
import random
import pickle
import sys

def weighted_random(values, weights):
	r = random.random()

	i = -1
	while r > 0:
		i += 1
		r -= weights[i]

	return values[i]

def set_controller_state(controller, cs):
	controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, cs.main_stick[0], cs.main_stick[1])
	controller.tilt_analog(melee.enums.Button.BUTTON_C, cs.c_stick[0], cs.c_stick[1])
	for b in cs.button:
		if (b != melee.enums.Button.BUTTON_START):
			if cs.button[b]:
				controller.press_button(b)
			else:
				controller.release_button(b)

	controller.press_shoulder(melee.enums.Button.BUTTON_L, cs.l_shoulder)
	controller.press_shoulder(melee.enums.Button.BUTTON_R, cs.r_shoulder)

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

data = loadData("./fox_fox_FD.pkl")

while True:
	gamestate = console.step()
	if gamestate is None:
		continue

	if console.processingtime * 1000 > 12:
		print("WARNING: Last frame took " + str(console.processingtime*1000) + "ms to process.")

	if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
		ps = gamestate.players[myPort]
		key = stringEnumerate(keyInfo(gamestate, myPort, opPort))
		vDict = valueFn(gamestate, myPort, opPort)
		vDict["value"] = np.array(vDict["value"])

		if (key in data["data"]):
			total = 0
			weights = []
			for value in data["data"][key]:
				w = np.linalg.norm(vDict["value"] - np.array(value["value"]))
				weights.append(w)
				total += w

			probabilities = np.array(weights) / total
			choice = weighted_random(data["data"][key], probabilities.tolist())
			set_controller_state(controller, choice["input"])


	else:
		melee.MenuHelper.menu_helper_simple(gamestate,
											controller,
											melee.Character.FOX,
											melee.Stage.FINAL_DESTINATION,
											costume=random.randint(1, 4),
											autostart=True,
											swag=False)
