import os
from comet_ml import Experiment
import pickle
import time

class Comet:
	def __init__(self):
		self.exp = Experiment(
			api_key=os.environ.get('COMET_API_KEY'), 
			project_name='ift6758-project',
			workspace='ds-team-9')
	
	def log_model(self, model, label, save_path):
		#save model to path
		timestamp = int(time.time())
		file_name = f'{save_path}_{timestamp}.pickle'
		pickle.dump(model, open(file_name, 'wb'))
		print(f"model saved: {file_name}")
		self.exp.log_model(label, file_name)

	def log_metrics(self, metrics):
		self.exp.log_metrics(metrics)

	def log_figure(self, name, figure, overwrite = False):
		self.exp.log_figure(figure_name = name,figure = figure, 
			overwrite = overwrite)

	def end(self):
		return self.exp.end()