from sapdi.serving.pymodel.exporter import PyExporter
from sapdi.serving.pymodel.predictor import AbstractPyModelPredictor

import os
import sys
import shutil

curr_dir = os.path.dirname(__file__)
sys.path.append(curr_dir)
path_to_python_example_model = "resources/models/py_models/model_1"

class ModelPredictor(AbstractPyModelPredictor):
    new_model = None

    def initialize(self, asset_files_path):
        print("Ensemble LR initialized")

        import pickle
        
        with open(asset_files_path + "/ensemble_lr.pkl", 'rb') as f:
            self.new_model = pickle.loads(f.read())


    def get_flower_name(self, res):
        if (res == 0):
            return "setosa"
        elif (res == 1):
            return "versicolor"
        elif (res == 2):
            return "virginica"
        else:
            return "N/A"

    def predict(self, input_arr):
        """
        input_arr is in the form of dictionary:
        Example:
        {
            "lr": <lr_result>,
            "knn": <knn_result>,
            "dt": <decision_tree_result>
        }
        """
        lr_arr = []
        knn_arr = []
        dt_arr = []

        for my_dict in input_arr:
            if my_dict.get("lr", None):
                lr_arr = my_dict["lr"]["output"]
                continue

            if my_dict.get("knn", None):
                knn_arr = my_dict["knn"]["output"]
                continue

            if my_dict.get("dt", None):
                dt_arr = my_dict["dt"]["output"]
                continue

        if not (len(lr_arr) == len(knn_arr) == len(dt_arr)):
            raise Exception("Wrong parameter input, different data length size")

        formatted_result = []

        for i in range(len(lr_arr)):
            lr_val = lr_arr[i]
            knn_val = knn_arr[i]
            dt_val = dt_arr[i]
            combined = [lr_val, knn_val, dt_val]
            formatted_result.append(combined)

        model_result = self.new_model.predict(formatted_result)
        final_result = []

        for res in model_result:
            final_result.append(self.get_flower_name(res))

        return {"output": final_result}


extracted_model_path = os.path.join(
    curr_dir, path_to_python_example_model, "ensemble_lr_iris_model"
)

exporter = PyExporter()
zipped_model_path = exporter.save_model(
    "ensemble_lr_iris_model",
    extracted_model_path,
    func=ModelPredictor(),
    asset_path_list=[
        os.path.join(
            curr_dir, path_to_python_example_model, "assets/ensemble_lr.pkl"
        )
    ],
    pip_dependency_file_path=os.path.join(
        curr_dir, path_to_python_example_model, "pip_dependencies.txt"
    )
)
shutil.unpack_archive(zipped_model_path, extract_dir=extracted_model_path)
