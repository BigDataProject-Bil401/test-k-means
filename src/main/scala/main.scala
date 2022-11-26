import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

object KmeansExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Test Kmeans")
    val sc = new SparkContext(conf)
    val v1 = Vectors.dense(Array(100000.0, 15000.0, 10.0))
    val v2 = Vectors.dense(Array(1500000.0, 30000.0, 2.0))
    val v3 = Vectors.dense(Array(1500000.0,  30000.0, 1.0))

    val rdd = sc.parallelize(List(v1, v2, v3))
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(rdd, numClusters, numIterations)
    sc.stop()

    val testSet = Vectors.dense(Array(100000.0, 15000.0, 10.0))
    val testSet2 = Vectors.dense(Array(1500000.0, 30000.0, 1.5))
    val samplePrediction = clusters.predict(testSet)
    val samplePrediction2 = clusters.predict(testSet2)

    println("cluster-centers -> ")
    clusters.clusterCenters.foreach(println)

    println(samplePrediction.toString())
    println(samplePrediction2.toString())
  }
}
