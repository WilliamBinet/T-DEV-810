from imageai.Prediction.Custom import CustomImagePrediction
import os
import glob

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "data/models/model_ex-095_acc-0.820312.h5"))
prediction.setJsonPath(os.path.join(execution_path, "data/json/model_class.json"))
prediction.loadModel(num_objects=2)

path = os.path.join(execution_path, "data/test/NORMAL/*.jpeg")

print (path)

files = glob.glob(path)

results_array = prediction.predictMultipleImages(files, result_count_per_image=2)

for each_result in results_array:
	predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
	for index in range(len(predictions)):
		print(predictions[index] , " : " , percentage_probabilities[index])
	print("-----------------------")
