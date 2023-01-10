import os
import comet_ml
import pickle
import time


ALREADY_USING_MESSAGE = "You are laready using the same model"
LOADING_MESSAGE = "Loading the model..."

class CometClient:
	dump_path = '/tmp/nhl_models/'

	def __init__(self, workspace, logger):
		self.workspace = 'ds-team-9'
		self.logger = logger
		self.api = comet_ml.API(api_key=os.getenv("COMET_API_KEY"))
		if not os.path.isdir(self.dump_path):
			os.makedirs(self.dump_path)

	def get_model(self, current_model, model_name, version):
		model_folder = os.path.join(self.dump_path, model_name, version)

		if os.path.isdir(model_folder):
			self.logger.info(ALREADY_USING_MESSAGE
				if current_model == model_name else LOADING_MESSAGE)
		else:
			try:
				self.logger.info(f"Downloading model[{model_name}:{version}] ...")
				self.api.download_registry_model(self.workspace, model_name, version,
				output_path = model_folder, expand = True)
			except comet_ml.exceptions.CometRestApiException as e:
				self.logger.error(f"Failure to download model - {model_name}:{version} from registry.")
				self.logger.info(f"Keeping the current loaded model - {current_model}")
				model_name = current_model
			else:
				self.logger.info("Model download successful!")

		model_folder = os.path.join(self.dump_path, model_name, version)
		model_file_name = os.listdir(model_folder)[0]
		model_path = os.path.join(model_folder, model_file_name)
		response = f'workspace: {self.workspace}, model_name: {model_name}, version: {version}'
		
		return model_path, response, model_name