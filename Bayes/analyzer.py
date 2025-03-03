from dataset_collector import *
import pickle
import melee

# """
with open("./pickles/compressed_testing.pkl", 'rb') as f:
	data = pickle.load(f)

for key in data["data"].keys():
	# print(key)
	for v in data["data"][key]:
		if (v["input"].button[melee.Button.BUTTON_X]):
			print(key)
# """