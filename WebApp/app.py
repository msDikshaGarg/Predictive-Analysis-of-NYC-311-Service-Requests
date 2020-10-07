import flask
from flask import Flask, render_template
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.sql.functions import dayofweek
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("proj").config("spark.some.config.option", "some-value").getOrCreate()

sc = spark.sparkContext

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Method to train model 
def model_train(zipcode, complaint, day):
	print ("Loading Data ...")
	data311 = spark.read.format("csv") .option("header", "true") .load(r"C:\Users\myste\Documents\MSCS\Spring 2020\Big Data\Project\Data\311_preprocessed\*.csv")
	infer_schema = "true"
	first_row_is_header = "true"
	delimiter = ","
	data311.registerTempTable("data311")
	data311 = data311.withColumn("ResTimeH", data311.Resolution_Time_Hours.cast('int'))
	data311 = data311.withColumn('day_of_week',dayofweek(data311['Created Date']))
	data311 = data311.withColumn("Zip", data311["Incident Zip"].cast('int'))
	data311 = data311.filter(data311.ResTimeH>0) 
	data311 = data311.filter(data311.ResTimeH<99) 
	bucketizer = Bucketizer(splits=[ 0,2,6,float('Inf') ],inputCol="ResTimeH", outputCol="categories")
	data311 = bucketizer.setHandleInvalid("keep").transform(data311)
	X=data311['Zip','Complaint_Type_Groups','day_of_week','categories']
	X = X.filter(X["Zip"]. isNotNull())
	X = X.filter(X["Complaint_Type_Groups"]. isNotNull())
	X = X.filter(X["day_of_week"]. isNotNull())

	stage_1 = StringIndexer(inputCol="Complaint_Type_Groups", outputCol="categoryIndex")
	stage_2 = OneHotEncoderEstimator(inputCols=["categoryIndex"],outputCols=["categoryVec"])
	stage_3 = VectorAssembler(inputCols=['Zip', 'day_of_week', 'categoryVec'],outputCol="features")
	stage_4 = StandardScaler().setInputCol("features").setOutputCol("Scaled_ip_features")
	stage_5 = LogisticRegression(labelCol="categories",featuresCol="Scaled_ip_features")
	# setup the pipeline
	pipeline = Pipeline(stages=[stage_1, stage_2, stage_3, stage_4, stage_5])
	# fit the pipeline model and transform the data as defined
	pipeline_model = pipeline.fit(X)

	zipcode = int(zipcode)
	day = int(day)
	input_variables = pd.DataFrame([[zipcode, complaint, day]], columns=['Zip', 'Complaint_Type_Groups', 'day_of_week'])
	input_variables = spark.createDataFrame(input_variables)

	transformed = pipeline_model.transform(input_variables)
	ans = transformed.select(collect_list('prediction')).first()[0]

	if (ans[0]==0.0):
		prediction="Your complaint will be resolved within 2 hours."
	elif (ans[0]==1.0):
		prediction="Your complaint will be resolved within 2-6 hours."
	else:
		prediction="Your complaint will be resolved after 6 hours"
	return prediction

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
	    # Just render the initial form, to get input
	    return(flask.render_template('main.html'))

	if flask.request.method == 'POST':
	    # Extract the input
	    day_vals = {"1" : "Monday", "2" : "Tuesday", "3" : "Wednesday", "4" : "Thursday", "5" : "Friday", "6" : "Saturday", "7" : "Sunday"}
	    zipcode = flask.request.form['zipcode']
	    complaint = flask.request.form['complaint']
	    day = flask.request.form['day']
	    for key, value in day_vals.items():
	        if key == day:
	            day_value=value

	    # Get the model's prediction
	    prediction = model_train(zipcode, complaint, day)

	    # Render the form again, but add in the prediction and remind user
	    # of the values they input before
	    return flask.render_template('main.html', original_input={'Zip Code':zipcode, 'Complaint':complaint, 'Day of the Week':day_value}, result=prediction,)

if __name__ == '__main__':
    app.run()