import os
import melee
import pickle

slippi_root = "/home/avighna/Slippi"

def ice_god_samus(gamestate):
	if len(gamestate.players.keys()) != 2:
		return -1

	samusPort = -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].character == melee.enums.Character.SAMUS):
			if ("ice" in gamestate.players[port].connectCode.lower()):
				return port
			else:
				if (samusPort == -1):
					samusPort = port
				else:
					return -1

	return samusPort

filefunctions = {"ice_god_samus": ice_god_samus}

file = "ice_god_samus.pkl"
fn = filefunctions[file.split('.')[0]]

data = {
	
}

if os.path.exists("./" + file):
	data = pickle.load(open("./" + file))

slippi_files = []
for root, _, files in os.walk(slippi_root):
    for file in sorted(files):
        slippi_files.append(os.path.join(root, file))

for slp_file in slippi_files:
    console = melee.Console(system="file", path=slp_file)
    console.connect()

    gamestate = console.step()
    print(slp_file)
    if (ice_god_samus(gamestate)):
	    while gamestate != None:


	        gamestate = console.step()
