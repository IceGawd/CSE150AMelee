import melee
import numpy as np
import os
import time

def getArray(point):
	return np.concatenate([point["value"].flatten(), point["input"].to_numpy().flatten()])

csButtonKeys = [
	melee.enums.Button.BUTTON_A,
	melee.enums.Button.BUTTON_B,
	melee.enums.Button.BUTTON_X,
	melee.enums.Button.BUTTON_Y,
	melee.enums.Button.BUTTON_Z,
	melee.enums.Button.BUTTON_L,
	melee.enums.Button.BUTTON_R,
	melee.enums.Button.BUTTON_START,
	melee.enums.Button.BUTTON_D_UP,
	melee.enums.Button.BUTTON_D_DOWN,
	melee.enums.Button.BUTTON_D_LEFT,
	melee.enums.Button.BUTTON_D_RIGHT
]

class PickleableControllerState(melee.controller.ControllerState):
	def __init__(self, controller_state=None, np_array=None):
		if np_array is not None:
			self.from_numpy(np_array)
		elif controller_state is not None:
			self.button = controller_state.button  # Dictionary {Button: bool}
			self.c_stick = controller_state.c_stick  # (x, y)
			self.l_shoulder = controller_state.l_shoulder  # float
			self.main_stick = controller_state.main_stick  # (x, y)
			self.r_shoulder = controller_state.r_shoulder  # float
		else:
			self.button = {}  # Default empty dictionary
			self.c_stick = (0.5, 0.5)  # Neutral
			self.l_shoulder = 0.0
			self.main_stick = (0.5, 0.5)  # Neutral
			self.r_shoulder = 0.0

	def __getstate__(self):
		state = {
			"button": self.button,
			"c_stick": self.c_stick,
			"l_shoulder": self.l_shoulder,
			"main_stick": self.main_stick,
			"r_shoulder": self.r_shoulder
		}
		return state

	def __setstate__(self, state):
		self.button = state["button"]
		self.c_stick = state["c_stick"]
		self.l_shoulder = state["l_shoulder"]
		self.main_stick = state["main_stick"]
		self.r_shoulder = state["r_shoulder"]

	def to_numpy(self):
		button_array = np.array([int(v) for v in self.button.values()])
		return np.concatenate([
			button_array,
			np.array(self.c_stick),
			np.array([self.l_shoulder]),
			np.array(self.main_stick),
			np.array([self.r_shoulder])
		])

	def from_numpy(self, np_array):
		global csButtonKeys

		# print([np_array[i] for i, button in enumerate(csButtonKeys)])

		button_size = len(csButtonKeys)
		self.button = {button: bool(np_array[i] > 5e-9) for i, button in enumerate(csButtonKeys)}
		self.c_stick = tuple(np_array[button_size:button_size + 2])
		self.l_shoulder = float(np_array[button_size + 2])
		self.main_stick = tuple(np_array[button_size + 3:button_size + 5])
		self.r_shoulder = float(np_array[button_size + 5])

def playerCharacter(gamestate, character, player):
	if len(gamestate.players.keys()) != 2:
		return -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].cpu_level == 0):
			if (gamestate.players[port].character == character):
				if (player in gamestate.players[port].connectCode.lower()):
					return port

	return -1

def colorCharacter(gamestate, character, color):
	if len(gamestate.players.keys()) != 2:
		return -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].cpu_level == 0):
			if (gamestate.players[port].character == character):
				if (gamestate.players[port].connectCode == "" and gamestate.players[port].costume == color):
					return port

	return -1

def me_you(gamestate, me, you):
	if len(gamestate.players.keys()) != 2:
		return -1

	samusPort = -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].character == me):
			if (gamestate.players[port].cpu_level == 0):
				if (samusPort == -1):
					samusPort = port
				else:
					return -1
		elif (gamestate.players[port].character != you):
			return -1

	return samusPort


def shunnash_fox(gamestate):
	return max(playerCharacter(gamestate, melee.Character.FOX, "simi"), colorCharacter(gamestate, melee.Character.FOX, 0), colorCharacter(gamestate, melee.Character.FOX, 1))

def ice_god_fox(gamestate):
	# print(playerCharacter(gamestate, melee.Character.FOX, "ice"))
	# print(colorCharacter(gamestate, melee.Character.FOX, 2))
	return max(playerCharacter(gamestate, melee.Character.FOX, "ice"), colorCharacter(gamestate, melee.Character.FOX, 2))

def ice_god_falco(gamestate):
	return max(playerCharacter(gamestate, melee.Character.FALCO, "ice"), colorCharacter(gamestate, melee.Character.FALCO, 2))

def ice_god_samus(gamestate):
	return playerCharacter(gamestate, melee.Character.SAMUS, "ice")

def samus_marth(gamestate):
	return me_you(gamestate, melee.Character.SAMUS, melee.Character.MARTH)

def falco_marth(gamestate):
	return me_you(gamestate, melee.Character.FALCO, melee.Character.MARTH)

def marth_fox(gamestate):
	return me_you(gamestate, melee.Character.MARTH, melee.Character.FOX)

def falco_falcon(gamestate):
	return me_you(gamestate, melee.Character.FALCO, melee.Character.CPTFALCON)

def fox_fox(gamestate):
	return me_you(gamestate, melee.Character.FOX, melee.Character.FOX)

def fox_falco(gamestate):
	return me_you(gamestate, melee.Character.FOX, melee.Character.FALCO)

def fox_fox_FD(gamestate):
	if len(gamestate.players.keys()) != 2:
		return -1

	if gamestate.stage != melee.Stage.FINAL_DESTINATION:
		return -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].character != melee.Character.FOX) or gamestate.players[port].cpu_level != 0:
			return -1

	return port


filefunctions = {
	"ice_god_samus": [ice_god_samus, False], 
	"ice_god_falco": [ice_god_falco, False], 
	"ice_god_fox": [ice_god_fox, False], 
	"shunnash_fox": [shunnash_fox, False], 
	"fox_fox_FD": [fox_fox_FD, False], 
	"samus_marth": [samus_marth, False], 
	"falco_falcon": [falco_falcon, False], 
	"falco_marth": [falco_marth, False], 
	"fox_falco": [fox_falco, False], 
	"fox_fox": [fox_fox, True],
	"testing": [marth_fox, False],
}

def keyInfo(gamestate, myPort, opPort):
	info = []
	info.append(gamestate.players[myPort].action)

	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		info.append(playerstate.character)
	info.append(gamestate.stage)
	return info

def stringEnumerate(info):
	s = ""
	for i in info:
		s += str(i).replace('.', '')
	return s

valueWeighting = np.array([7, 3, 7, 5, 1, 5, 5, 3, 3, 3, 3, 3, 1, 1, 1, 1])
valueWeighting = np.concatenate([3 * valueWeighting, valueWeighting])

def valueFn(gamestate, myPort, opPort):
	value = []
	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		value.append(float(playerstate.off_stage))
		value.append(float(playerstate.facing))
		value.append(float(playerstate.on_ground))
		value.append(float(playerstate.jumps_left))
		value.append(float(playerstate.stock))
		value.append(float(playerstate.position.x))
		value.append(float(playerstate.position.y))
		value.append(float(playerstate.speed_air_x_self))
		value.append(float(playerstate.speed_ground_x_self))
		value.append(float(playerstate.speed_x_attack))
		value.append(float(playerstate.speed_y_attack))
		value.append(float(playerstate.speed_y_self))
		value.append(float(playerstate.percent))
		value.append(float(playerstate.hitstun_frames_left))
		value.append(float(playerstate.invulnerability_left))
		value.append(float(playerstate.shield_strength))

	vDict = {}
	vDict["input"] = PickleableControllerState(gamestate.players[myPort].controller_state)
	vDict["value"] = value

	# print(list(vDict["input"].button.keys()))
	# print(vDict["input"].to_numpy())
	# print(vDict["value"])
	# print(vDict["input"].to_numpy().shape[0])
	# print(len(vDict["value"]))

	return vDict

def otherPort(gamestate, myPort):
	for port in gamestate.players.keys():
		if myPort != port:
			return port

	raise Exception("How?")

def loopThrough(addData, saveData, loadData, filesFrom=["2025"], savefile="ice_god_fox"):
	slippi_root = "/home/avighna/Slippi"
	# slippi_root = "/home/avighna/Documents/python/CSE150AMelee/Bayes"

	fn = filefunctions[savefile][0]

	data = {
		"last_file": None,
		"data": {}
	}

	if os.path.exists("./pickles/" + savefile + ".pkl"):
		data = loadData(savefile)

	slippi_files = []
	for root, _, files in os.walk(slippi_root):
		for file in files:
			if (file[-4:] == ".slp"):
				slippi_files.append(os.path.join(root, file))

	slippi_files.sort(key = lambda x: x.split('/')[-1])

	lastSaved = time.time()
	reading = data["last_file"] == None

	for slp_file in slippi_files:
		if reading:
			if (max([(f in slp_file) for f in filesFrom])):
				console = melee.Console(system="file", path=slp_file)
				console.connect()

				try:
					gamestate = console.step()
				except:
					print(slp_file + " (Weird???)")
					continue

				if (gamestate == None):
					print(slp_file + " (Zero/One Frame Game)")
					continue

				port = fn(gamestate)
				if (port != -1):
					port2 = otherPort(gamestate, port)
					print(slp_file + " (Valid)")

					data["data"]["prev"] = {}

					while gamestate != None:
						addData(data, gamestate, port, port2)

						if filefunctions[savefile][1]:
							addData(data, gamestate, port2, port)

						gamestate = console.step()
				else:
					print(slp_file + " (Invalid)")

				data["last_file"] = slp_file

				if time.time() - lastSaved > 60:						# Save after one minute of processing
					saveData(savefile, data)
					lastSaved = time.time()

				console.stop()
		else:
			reading = slp_file == data["last_file"]

	saveData(savefile, data)

	return data

def normalize(point, minVal, maxVal):
	point = (np.array(point)) / (maxVal - minVal)
	return np.nan_to_num(point, nan=0.5, posinf=1, neginf=0)

def data_normalization(data):
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
			d["value"] = normalize(d["value"], minVal, maxVal)

	return minVal, maxVal

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