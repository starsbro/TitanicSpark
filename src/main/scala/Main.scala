//Importing required libraries

import org.apache.spark.sql.types.{StructType, StringType, IntegerType, FloatType}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.GBTClassifier


object Main {
  def main(args: Array[String]): Unit = {
    println("Hello world!")

    // Question 1 Exploratory Data Analysis
    // Follow up on the previous spark assignment 1 and explain a few statistics. (20 pts)
    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("Titanic ML Model")
      .master("local") // Use "local" for local testing, or adjust for cluster
      .getOrCreate()

    // Schema for importing training and testing datasets
    // Training data includes Survived column, while test data doesn't.
    val dataScheme_train = (new StructType)
      .add("PassengerId", IntegerType)
      .add("Survived", IntegerType)
      .add("Pclass", IntegerType)
      .add("Name", StringType)
      .add("Sex", StringType)
      .add("Age", FloatType)
      .add("SibSp", IntegerType)
      .add("Parch", IntegerType)
      .add("Ticket", StringType)
      .add("Fare", FloatType)
      .add("Cabin", StringType)
      .add("Embarked", StringType)

    val dataScheme_test = (new StructType)
      .add("PassengerId", IntegerType)
      .add("Pclass", IntegerType)
      .add("Name", StringType)
      .add("Sex", StringType)
      .add("Age", FloatType)
      .add("SibSp", IntegerType)
      .add("Parch", IntegerType)
      .add("Ticket", StringType)
      .add("Fare", FloatType)
      .add("Cabin", StringType)
      .add("Embarked", StringType)

    // Reading data using SparkSession
    val df_train = spark.read.format("csv").option("header", "true").schema(dataScheme_train).load("./src/main/resources/train.csv")
    val df_test = spark.read.format("csv").option("header", "true").schema(dataScheme_test).load("./src/main/resources/test.csv")

    // Create table views for training and testing (Optional, for SQL queries)
    df_train.createOrReplaceTempView("df_train")
    df_test.createOrReplaceTempView("df_test")

    // Show some data from training and testing dataframes for verification
    df_train.describe("Pclass","Age","SibSp", "Parch", "Fare").show()
    df_train.show(5)
    df_test.show(5)

    //SQL queries
    spark.sql("select Survived,count(*) from df_train group by Survived").show()
    spark.sql("Select Sex, Survived, count(*) from df_train group by Sex, Survived").show()
    spark.sql("Select Pclass, Survived, count(*) from df_train group by Pclass, Survived").show()

    //Calculate avg Age to fill null values for training
    val AvgAge = df_train.select("Age").agg(avg("Age")).collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }
    println("df_train Average age is ", AvgAge)

    //Calculate avg Fare for filling gaps in df_train
    val AvgFare = df_train.select("Fare").agg(avg("Fare")).collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }


    //Calculate avg Age to fill null values for df_test
    val AvgAge_test = df_test.select("Age").agg(avg("Age")).collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    //Calculate avg fail for filling gaps in df_test
    val AvgFare_test = df_test.select("Fare").agg(avg("Fare")).collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    //for training
    val embarked: String => String = {
      case "" => "S"
      case null => "S"
      case a => a
    }
    val embarkedUDF = udf(embarked)

    //for test
    val embarked_test: String => String = {
      case "" => "S"
      case null => "S"
      case a => a
    }
    val embarkedUDF_test = udf(embarked_test)


    // Question 2 Feature Engineering - Create new attributes that may be derived from the existing attributes.
    //  This may include removing certain columns in the dataset. (30 pts)

    //Create new attributes: FamilySize
    val df_trainWithFamilySize = df_train.withColumn("FamilySize", col("SibSp") + col("Parch"))

    // Show the new DataFrame with the "FamilySize" column
    df_trainWithFamilySize.describe("FamilySize").show()
    df_trainWithFamilySize.show(5)

    //Register the DataFrame as a temporary view
    df_trainWithFamilySize.createOrReplaceTempView("df_trainWithFamilySize")
    spark.sql("Select Sex, Survived, FamilySize, count(*) from df_trainWithFamilySize group by Sex, Survived, FamilySize").show()


    // Question 3 Prediction - Use the train.csv to train a Machine Learning model
    // Filling null values with avg values for training dataset
    val imputeddf = df_train.na.fill(Map("Fare" -> AvgFare, "Age" -> AvgAge))
    val imputeddf2 = imputeddf.withColumn("Embarked", embarkedUDF(imputeddf.col("Embarked")))

    //Splitting training data into training and validation
    val Array(trainingData, validationData) = imputeddf2.randomSplit(Array(0.7, 0.3))

    //Filling null values with avg values for test dataset
    val imputeddf_test = df_test.na.fill(Map("Fare" -> AvgFare_test, "Age" -> AvgAge_test))
    val imputeddf2_test = imputeddf_test.withColumn("Embarked", embarkedUDF_test(imputeddf_test.col("Embarked")))

    //Dropping Cabin feature as it has so many null values
    val df1_train = trainingData.drop("Cabin")
    val df1_test = imputeddf2_test.drop("Cabin")

    //Print schema for train and test
    df1_train.printSchema()
    df1_test.printSchema()

    //Indexing categorical features
    val catFeatColNames = Seq("Pclass", "Sex", "Embarked")
    val stringIndexers = catFeatColNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
        .fit(trainingData)
    }

    //Indexing target feature
    val labelIndexer = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("SurvivedIndexed")
      .fit(trainingData)

    //Assembling features into one vector
    val numFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")
    val idxdCatFeatColName = catFeatColNames.map(_ + "Indexed")
    val allIdxdFeatColNames = numFeatColNames ++ idxdCatFeatColName
    val assembler = new VectorAssembler()
      .setInputCols(Array(allIdxdFeatColNames: _*))
      .setOutputCol("Features")

    //RandomForest classifier
    val randomForest = new RandomForestClassifier()
      .setLabelCol("SurvivedIndexed")
      .setFeaturesCol("Features")

    //Retrieving original labels
    // Assuming labelIndexer is configured correctly to output an array of strings
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))

    //Creating pipeline
    val pipeline = new Pipeline().setStages(
      (stringIndexers :+ labelIndexer :+ assembler :+ randomForest :+ labelConverter).toArray
    )

    //Selecting best model
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxBins, Array(25, 28, 31))
      .addGrid(randomForest.maxDepth, Array(4, 6, 8))
      .addGrid(randomForest.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("SurvivedIndexed")
      .setMetricName("areaUnderPR")

    //Cross validator with 10-fold
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    //Fitting model using cross validation
    val crossValidatorModel = cv.fit(trainingData)

    //predictions on validation data
    val predictions = crossValidatorModel.transform(validationData)

    //Accuracy
    val accuracy = evaluator.evaluate(predictions)
    println("Test1 accuracy= " + accuracy)
    println("Test1 Error DT= " + (1.0 - accuracy))

    //predicting on test data
    val predictions_test = crossValidatorModel.transform(df1_test)


    predictions_test
      .withColumn("Survived", col("predictedLabel"))
      .select("PassengerId", "Survived")
      .coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .save("./src/main/resources/output")

    spark.sql("select * from output").collect.foreach(println)

    ////Implementing Gradient boosted tree
    val gbt = new GBTClassifier()
      .setLabelCol("SurvivedIndexed")
      .setFeaturesCol("Features")
      .setMaxIter(10)

    //Creating pipeline
    val pipeline1 = new Pipeline().setStages(
      (stringIndexers :+ labelIndexer :+ assembler :+ gbt :+ labelConverter).toArray)

    val model = pipeline1.fit(trainingData)

    val predictions_model = model.transform(validationData)

    val evaluator_result = new BinaryClassificationEvaluator()
      .setLabelCol("SurvivedIndexed")
      .setMetricName("areaUnderPR")

    val accuracy_result = evaluator_result.evaluate(predictions_model)
    println("Test2 Error = " + (1.0 - accuracy_result))
  }
}