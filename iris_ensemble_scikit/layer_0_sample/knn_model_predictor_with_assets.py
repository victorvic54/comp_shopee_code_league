from sapdi.serving.pymodel.exporter import PyExporter
from sapdi.serving.pymodel.predictor import AbstractPyModelPredictor

import os
import sys
import shutil

curr_dir = os.path.dirname(__file__)
sys.path.append(curr_dir)
path_to_python_example_model = "resources/models/py_models/model_1"

class ModelPredictor(AbstractPyModelPredictor):
    stack_model = None

    def initialize(self, asset_files_path):
        print("KNN object initialized")

        import pickle
        
        with open(asset_files_path + "/knn.pkl", 'rb') as f:
            self.new_model = pickle.loads(f.read())


    def predict(self, input_arr):
        if ("input" not in input_arr):
            raise Exception("Invalid predict function body")

        """
        input_arr is in the form of list:
        Example:
        {"input": [[6,2,4,2]]}
        """

        model_result = self.stack_model.predict(input_arr["input"])

        return {"knn": {"output": model_result.tolist}}


extracted_model_path = os.path.join(
    curr_dir, path_to_python_example_model, "knn_iris_model"
)

exporter = PyExporter()
zipped_model_path = exporter.save_model(
    "knn_iris_model",
    extracted_model_path,
    func=ModelPredictor(),
    pip_dependency_file_path=os.path.join(
        curr_dir, path_to_python_example_model, "pip_dependencies.txt"
    )
)
shutil.unpack_archive(zipped_model_path, extract_dir=extracted_model_path)
