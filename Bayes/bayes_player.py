import melee
import signal
from dataset_collector import *
import numpy as np
import random
import pickle
import sys
import time

savefile = "ice_god_fox.pkl"
# connect_code = "CELL#835"
connect_code = ""

character_stage = {
	"samus_marth": [melee.Character.SAMUS, melee.Stage.RANDOM_STAGE], 
	"fox_fox": [melee.Character.FOX, melee.Stage.RANDOM_STAGE], 
	"ice_god_fox": [melee.Character.FOX, melee.Stage.RANDOM_STAGE], 
	"falco_falcon": [melee.Character.FALCO, melee.Stage.RANDOM_STAGE]
}

character, stage = character_stage[savefile.split('.')[0]]
costume = random.randint(0, 4)

data = loadData("./" + savefile)

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
		if (b != melee.Button.BUTTON_START):
			if cs.button[b]:
				controller.press_button(b)
			else:
				controller.release_button(b)

	controller.press_shoulder(melee.Button.BUTTON_L, cs.l_shoulder)
	controller.press_shoulder(melee.Button.BUTTON_R, cs.r_shoulder)

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
	start = time.time()

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
		vDict = valueFn(gamestate, myPort, opPort)
		vDict["value"] = np.array(vDict["value"])

		if (key in data["data"]):
			print(len(data["data"][key]))

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
											character,
											stage,
											connect_code,
											costume=costume,
											autostart=True,
											swag=False)

	# end = time.time()

	# print("Frame: " + str(end - start))