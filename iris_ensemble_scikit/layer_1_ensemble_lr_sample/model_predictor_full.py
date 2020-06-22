class ModelPredictor(AbstractPyModelPredictor):
    new_model = None

    def initialize(self, asset_files_path):
        print("Ensemble LR initialized")
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics
        import pandas as pd

        model1 = DecisionTreeClassifier(random_state=1)
        model2 = KNeighborsClassifier()
        model3 = LogisticRegression(random_state=42, max_iter=300)
        iris = datasets.load_iris()

        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        model3.fit(X_train, y_train)
        
        # accuracy calculation
        y_pred_1 = pd.DataFrame(model1.predict(X_test))
        y_pred_2 = pd.DataFrame(model2.predict(X_test))
        y_pred_3 = pd.DataFrame(model3.predict(X_test))
        
        print("Accuracy of DT:", metrics.accuracy_score(y_test, y_pred_1))
        print("Accuracy of KNN:", metrics.accuracy_score(y_test, y_pred_2))
        print("Accuracy of LR:", metrics.accuracy_score(y_test, y_pred_3))
        
        df = pd.concat([y_pred_1, y_pred_2, y_pred_3], axis=1)
        self.new_model = LogisticRegression(random_state=1)
        self.new_model.fit(df, y_test)
    
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
