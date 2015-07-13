name := "dl4j-test"

version := "0.1"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.nd4j" % "nd4j-jblas" % "0.0.3.5.5.5",
  "org.nd4j" % "canova-parent" % "0.0.0.4",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.0.3.3.4.alpha2"
)
