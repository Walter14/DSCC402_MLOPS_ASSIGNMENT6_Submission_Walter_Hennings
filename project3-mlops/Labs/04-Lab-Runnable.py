# Databricks notebook source
# TODO
#Create 3 widgets for parameter passing into the notebook:
#  - n_estimators with a default of 100
#  - learning_rate with a default of .1
#  - max_depth with a default of 1 
#Note that only strings can be used for widgets

dbutils.widgets.text("n_estimators", "100")
dbutils.widgets.text("learning_rate", ".1")
dbutils.widgets.text("max_depth", "1")

# COMMAND ----------

# TODO
#Read from the widgets to create 3 variables.  Be sure to cast the values to numeric types
n_estimators = dbutils.widgets.get("n_estimators")
learning_rate = dbutils.widgets.get("learning_rate")
max_depth = dbutils.widgets.get("max_depth")

# COMMAND ----------

# TODO
#Train and log the results from a model.  Try using Gradient Boosted Trees
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
from sklearn.datasets import make_regression
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

with mlflow.start_run(run_name="Gradient Boosting") as run:
  # Create model, train it, and create predictions
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    predictions = gb.predict(X_test)
  
  # Log model
    mlflow.sklearn.log_model(gb, "random-forest-model")
  
  # Create metrics
    mse = mean_squared_error(y_test, predictions)
    print(f"mse: {mse}")
  
  # Log metrics
    mlflow.log_metric("mse", mse)
  
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    artifactURI = mlflow.get_artifact_uri()
    predictions_output_path = artifactURI + "/predictions.csv"
  
    print(f"Inside MLflow Run with run_id `{runID}` and experiment_id `{experimentID}`")

# COMMAND ----------

# TODO
#Report the model output path to the parent notebook
import json


dbutils.notebook.exit(json.dumps({
  "status": "OK",
  "run_id": runID,
  "experiment_id": experimentID,
  "model_output_path": predictions_output_path
}))


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
