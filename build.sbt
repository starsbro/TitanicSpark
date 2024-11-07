ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.9"

lazy val root = (project in file("."))
  .settings(
    name := "TitanicSpark"
  )

val sparkVersion ="3.2.1"

libraryDependencies += "org.apache.spark"%%"spark-core"% sparkVersion

libraryDependencies +="org.apache.spark"%%"spark-sql"% sparkVersion

libraryDependencies +="org.apache.spark"%%"spark-mllib" % sparkVersion