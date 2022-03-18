import os, re
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_LEVEL = "99"

df_hH_EN_all = pd.DataFrame()
df_hH_ZH_all = pd.DataFrame()

for ipc in ["A","B","C","D","E","F","G","H"]:

	#df_hH = pd.read_csv("hH.csv", sep="\t", names=["section", "term", "hyper"], dtype=object).drop_duplicates()
	df_hH = pd.read_csv("./pairs/hH_ipc{}_{}_lower.csv".format(str(TARGET_LEVEL), ipc), header=None, sep=',', dtype=object, names=['hyponym','hyperonym']).drop_duplicates()

	#r'"?predict hypernym: <[A-H]> [^\n]+([a-z]{2,}"?,[^\n]+|[^\n]+,"?[a-z]{2,})[^\n]+\n'  >>>>>>  ''

	df_hH_EN = pd.DataFrame()
	df_hH_EN['hyponym'] = df_hH['hyponym'].astype(str).apply(lambda x: re.sub(r"\t+([^\n]+)","",x.lower()))
	df_hH_EN['hyperonym'] = df_hH['hyperonym'].astype(str).apply(lambda x: re.sub(r"\t+([^\n]+)","",x.lower()))
	df_hH_EN['section'] = ipc #df_hH['section'].astype(str)

	df_hH_ZH = pd.DataFrame()
	df_hH_ZH['hyponym'] = df_hH['hyponym'].astype(str).apply(lambda x: re.sub(r"([^\n]+)\t+","",x.lower()))
	df_hH_ZH['hyperonym'] = df_hH['hyperonym'].astype(str).apply(lambda x: re.sub(r"([^\n]+)\t+","",x.lower()))
	df_hH_ZH['section'] = ipc #df_hH['section'].astype(str)

	df_hH_EN_all = pd.concat([df_hH_EN_all, df_hH_EN], axis=0)
	df_hH_ZH_all = pd.concat([df_hH_ZH_all, df_hH_ZH], axis=0)

print(df_hH_EN_all.shape)
print(df_hH_ZH_all.shape)

"""
df_Hh_EN = df_hH_EN_all[["section", "hyperonym", "hyponym"]]
df_Hh_ZH = df_hH_ZH_all[["section", "hyperonym", "hyponym"]]

dict_df = { "hH_EN": df_hH_EN_all, 
			"Hh_EN": df_Hh_EN, 
			"hH_ZH": df_hH_ZH_all, 
			"Hh_ZH": df_Hh_ZH }
"""

dict_df = { "hH_EN": df_hH_EN_all, 
			"hH_ZH": df_hH_ZH_all }

for (name,data) in dict_df.items():
	print(name)
	
	# create train test data for hH
	data['source_text'] = ('predict hypernym: <' + data['section'] + '> ' + data['hyponym']).astype(str)
	data['target_text'] = data['hyperonym'].astype(str)
	# split 
	data = data[['source_text', 'target_text']].sample(frac=1)
	train, test = train_test_split(data, test_size=0.2)
	print(train.shape)
	print(test.shape)
	# save
	#print("./data/train_{}.csv".format(name))
	train.to_csv("./data/train_{}_ipc{}.csv".format(name,TARGET_LEVEL), index=False)
	#print("./data/test_{}.csv".format(name))
	test.to_csv("./data/test_{}_ipc{}.csv".format(name,TARGET_LEVEL), index=False)
"""

df_hH_EN_ZH = pd.concat([df_hH_EN_all, df_hH_ZH_all], axis=1)

_, manual_eval = train_test_split(df_hH_EN_ZH, test_size=500, random_state=100)
#train=df_unique.sample(frac=0.8,random_state=200)
print(manual_eval.shape)
#manual_eval

manual_eval.to_csv("./pairs/hH_random_ipc{}_500.csv".format(str(TARGET_LEVEL)), index=False)
"""





