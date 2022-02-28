import argparse
from tensorboard.backend.event_processing import event_accumulator as ea


from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
import os 
import pandas as pd
import numpy as np

sns.set(style="darkgrid")
sns.set_context("paper")

def plot_vanilla(df):
    sns.lineplot(x="episode", y="reward", hue="algo", data=df)
    plt.title("some loss")
    plt.show()

def tensor2np(params, min_len):
	''' beautify tf log
		Use better library (seaborn) to plot tf event file'''

	log_path = params['logdir']
	X = []
	Y = []
	dir_path = os.listdir(log_path)
	for algo in dir_path:
		print(algo)
		Y_algo = []
		t_path = os.listdir(log_path + algo)
		for t in t_path:
			path = log_path + algo + '/' + t
			print(path)

			acc = ea.EventAccumulator(path)
			acc.Reload()

			scalar_list = acc.Tags()['scalars']
			x_list = []
			y_list = []
			x_list_raw = []
			y_list_raw = []
			for tag in scalar_list:
				if tag != "train/reward":
					continue
				x = [int(s.step) for s in acc.Scalars(tag)]
				y = [s.value for s in acc.Scalars(tag)]

				# # smooth curve
				# x_ = []
				# y_ = []
				# for i in range(0, len(x), smooth_space):
				# 	x_.append(x[i])
				# 	y_.append(sum(y[i:i+smooth_space]) / float(smooth_space))    
				# x_.append(x[-1])
				# y_.append(y[-1])
				# x_list.append(x_)
				# y_list.append(y_)

				# raw curve
				x_list_raw.append(x)
				y_list_raw.append(y)

				X.append(np.array(x_list_raw)[0][:min_len])
				Y_algo.append(np.array(y_list_raw)[0][:min_len])
		Y.append(np.array(Y_algo))
	return Y

def np2pd(data):
	label = ['Fixed-input','trans_set','DeepSet', 'lstm']
	df = []
	for i in range(len(data)):
		df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
		df[i]['algo']= label[i] 

	df = pd.concat(df) # 合并
	return df

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', default='./plot/', type=str, help='logdir to event file')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	Y = tensor2np(params,300)
	pd = np2pd(Y)
	plot_vanilla(pd)
