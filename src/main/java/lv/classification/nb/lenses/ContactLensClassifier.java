package lv.classification.nb.lenses;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import edu.campus02.iwi.spark.env.WinConfig;

public class ContactLensClassifier {

	public static void main(String[] args) {

		WinConfig.setupEnv();

		// TODO: 1) set spark config and init spark session
		SparkConf cnf = new SparkConf()
				.setMaster("local")
				.setAppName(ContactLensClassifier.class.getName());
		
		SparkSession spark = SparkSession.builder()
				.config(cnf).getOrCreate();
		
		// TODO: 2) read raw data as csv with corresponding options
		Dataset<Row> raw = spark.read()
				.option("header", true) //CSV datei hat Kopfzeile
				.option("delimiter", ";") //Trennzeichen ist ‘;’
				.option("inferSchema", true) //Schema wirdermittelt
				.csv("data/input/lv/lenses/contact-lenses.csv");
		
		raw.printSchema();
		raw.show();
		
		// TODO: 3) combine the separate columns as feature vector
		// all f0..f8 => 9 columns in total
		
		VectorAssembler toVec = new VectorAssembler()
				.setInputCols(Arrays.copyOf(raw.columns(), 9))
				.setOutputCol("features");
		
		// TODO: 4) transform the raw data
		
		Dataset<Row> data = toVec.transform(raw);
		
		// TODO: 5) train a NaiveBayes model on the data
		
		NaiveBayesModel nbm = new NaiveBayes()
				.setSmoothing(1.0)
				.fit(data);
		
		// TODO: 6) predict a single hard-coded feature vector
		
		Vector demoVec = Vectors.dense(
				new double[] {1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0}
		);
		
		System.out.println("demo vector = "+ demoVec);
		//result of prediction should be 1.0 => Label (Class)1
		double singlePrediction = nbm.predict(demoVec);
		System.out.println("prediction = "+ singlePrediction);
		
		// TODO: 7) for demo only -> make predictions for all training vectors
		
		Dataset<Row> predictions = nbm.transform(data)
				.selectExpr("label","prediction",
						"IF(label=prediction,'OK','WRONG') AS result");
		
		predictions.show(25,false);
		
		// TODO: 8) compute accuracy for complete dataset
		
		MulticlassClassificationEvaluator evaluator =
				new MulticlassClassificationEvaluator()
				.setMetricName("accuracy");
		
		double accuracy = evaluator.evaluate(predictions);
		System.out.println("test data accuracy = "+ accuracy);
		
		// TODO: 9) close spark session
		
		spark.close();

	}

}