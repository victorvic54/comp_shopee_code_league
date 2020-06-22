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
        print("Object initialized")

        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

        self.stack_model = KNeighborsClassifier()
        self.stack_model.fit(X_train, y_train)

        # accuracy calculation
        y_pred = self.stack_model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        print("KNN Classifier accuracy level 0:", accuracy)

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
