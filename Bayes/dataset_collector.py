import os
import melee
import pickle
import time
from sklearn.cluster import KMeans

class PickleableControllerState(melee.controller.ControllerState):
	def __init__(self, controller_state):
		self.button = controller_state.button
		self.c_stick = controller_state.c_stick
		self.l_shoulder = controller_state.l_shoulder
		self.main_stick = controller_state.main_stick
		self.r_shoulder = controller_state.r_shoulder
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

def ice_god_samus(gamestate):
	return playerCharacter(gamestate, melee.Character.SAMUS, "ice")

def samus_marth(gamestate):
	return me_you(gamestate, melee.Character.SAMUS, melee.Character.MARTH)

def falco_falcon(gamestate):
	return me_you(gamestate, melee.Character.FALCO, melee.Character.CPTFALCON)

def fox_fox(gamestate):
	return me_you(gamestate, melee.Character.FOX, melee.Character.FOX)

def fox_fox_FD(gamestate):
	if len(gamestate.players.keys()) != 2:
		return -1

	if gamestate.stage != melee.Stage.FINAL_DESTINATION:
		return -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].character != melee.Character.FOX):
			return -1

	return port

def saveData(file, data):
	# print(data)
	print("DON'T QUIT Saving...")
	with open(file, 'wb') as f:
		pickle.dump(data, f)
	print("Saved!")

def otherPort(gamestate, myPort):
	for port in gamestate.players.keys():
		if myPort != port:
			return port

	raise Exception("How?")

def keyInfo(gamestate, myPort, opPort):
	info = []
	info.append(gamestate.players[myPort].action)

	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		info.append(playerstate.character)
		# info.append(playerstate.off_stage)
	info.append(gamestate.stage)
	return info

def valueFn(gamestate, myPort, opPort):
	value = []
	for port, EV in zip([myPort, opPort], [2, 1]):
		playerstate = gamestate.players[port]
		value.append(EV * float(playerstate.facing) * 50)
		value.append(EV * float(playerstate.jumps_left) * 20)
		value.append(EV * float(playerstate.stock) * 10)
		value.append(EV * float(playerstate.position.x * 5))
		value.append(EV * float(playerstate.position.y * 5))
		value.append(EV * float(playerstate.speed_air_x_self * 2))
		value.append(EV * float(playerstate.speed_ground_x_self * 2))
		value.append(EV * float(playerstate.speed_x_attack * 2))
		value.append(EV * float(playerstate.speed_y_attack * 2))
		value.append(EV * float(playerstate.speed_y_self * 2))
		value.append(EV * float(playerstate.percent))
		value.append(EV * float(playerstate.hitstun_frames_left))
		value.append(EV * float(playerstate.invulnerability_left))
		value.append(EV * float(playerstate.shield_strength))

	vDict = {}
	vDict["input"] = PickleableControllerState(gamestate.players[myPort].controller_state)
	vDict["value"] = value

	return vDict

def stringEnumerate(info):
	s = ""
	for i in info:
		s += str(i)
	return s

def addData(data, gamestate, myPort, opPort):
	key = stringEnumerate(keyInfo(gamestate, myPort, opPort))
	vDict = valueFn(gamestate, myPort, opPort)

	if key in data["data"]:
		data["data"][key].append(vDict)
	else:
		data["data"][key] = [vDict]

def loadData(savefile):
	print("Loading data...")
	with open("./" + savefile, "rb") as file:
	    return pickle.load(file)
	print("Data Loaded!")

if __name__ == "__main__":
	slippi_root = "/home/avighna/Slippi"

	filefunctions = {
		"ice_god_samus": [ice_god_samus, False], 
		"ice_god_fox": [ice_god_fox, False], 
		"shunnash_fox": [shunnash_fox, False], 
		"fox_fox_FD": [fox_fox_FD, False], 
		"samus_marth": [samus_marth, False], 
		"falco_falcon": [falco_falcon, False], 
		"fox_fox": [fox_fox, True],
	}

	savefile = "ice_god_fox.pkl"
	fn = filefunctions[savefile.split('.')[0]][0]

	data = {
		"last_file": None,
		"data": {}
	}

	if os.path.exists("./" + savefile):
		data = loadData(savefile)

	slippi_files = []
	for root, _, files in os.walk(slippi_root):
		for file in files:
			slippi_files.append(os.path.join(root, file))

	slippi_files.sort(key = lambda x: x.split('/')[-1])

	lastSaved = time.time()
	reading = data["last_file"] == None

	for slp_file in slippi_files:
		if reading:
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
				while gamestate != None:
					addData(data, gamestate, port, port2)
					gamestate = console.step()
			else:
				print(slp_file + " (Invalid)")

			data["last_file"] = slp_file

			if time.time() - lastSaved > 30:						# Save after one minute of processing
				saveData(savefile, data)
				lastSaved = time.time()

			console.stop()
		else:
			reading = slp_file == data["last_file"]

	saveData(savefile, data)