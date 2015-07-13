package core

import java.util.Random
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.conf.`override`.ConfOverride
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import scala.collection.JavaConverters._

object Main {

  def main(args: Array[String]) {
    // Customizing params
    Nd4j.MAX_SLICES_TO_PRINT = -1
    Nd4j.MAX_ELEMENTS_PER_SLICE = -1

    val numRows = 4
    val numColumns = 1
    val outputNum = 3
    val numSamples = 150
    val batchSize = 150
    val iterations = 25
    val splitTrainNum = (batchSize * 0.8).toInt
    val seed = 123
    val listenerFreq = iterations - 1

    Nd4j.getRandom().setSeed(seed)

    println("Load data....")
    val iter = new IrisDataSetIterator(batchSize, numSamples)
    val next = iter.next()
    next.normalizeZeroMeanZeroUnitVariance()

    println("Split data....")
    val testAndTrain = next.splitTestAndTrain(splitTrainNum)
    val train = testAndTrain.getTrain()
    val test = testAndTrain.getTest()

    println("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .layer(new RBM()) // NN layer type
      .nIn(numRows * numColumns) // # input nodes
      .nOut(outputNum) // # output nodes
      .visibleUnit(RBM.VisibleUnit.GAUSSIAN) // Gaussian transformation visible layer
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED) // Rectified Linear transformation visible layer
      .iterations(iterations) // # training iterations predict/classify & backprop
      .weightInit(WeightInit.NORMALIZED) // Weight initialization method
      .activationFunction("relu") // Activation function type
      .k(1) // # contrastive divergence iterations
      .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
      .learningRate(1e-3f) // Optimization step size
      .optimizationAlgo(OptimizationAlgorithm.LBFGS) // Backprop method (calculate the gradients)
      .constrainGradientToUnitNorm(true)
      .regularization(true)
      .l2(2e-4)
      .momentum(0.9)
      .list(2) // # NN layers (does not count input layer)
      .hiddenLayerSizes(12) // # fully connected hidden layer nodes. Add list if multiple layers.
      .`override`(1, new ConfOverride() {
        def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder) {
          builder.activationFunction("softmax");
          builder.layer(new OutputLayer());
          builder.lossFunction(LossFunctions.LossFunction.MCXENT);
          builder.optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
        }
      })
      .build()
    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(List[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

    println("Train model....")
    model.fit(train)

    println("Evaluate weights....")
    model.getLayers().foreach { layer =>
      val w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
      println("Weights: " + w)
    }

    println("Evaluate model....")
    val eval = new Evaluation()
    val output = model.output(test.getFeatureMatrix());

    (0 until output.rows()).foreach { i =>
      val actual = train.getLabels().getRow(i).toString().trim()
      val predicted = output.getRow(i).toString().trim()
      println("actual " + actual + " vs predicted " + predicted)
    }

    eval.eval(test.getLabels(), output)
    println(eval.stats())
    println("****************Example finished********************")

  }
}
