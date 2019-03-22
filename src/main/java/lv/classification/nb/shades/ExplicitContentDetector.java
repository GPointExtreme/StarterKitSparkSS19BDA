package lv.classification.nb.shades;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import edu.campus02.iwi.spark.env.WinConfig;
import spark.exercise.utils.PreprocessUtils;

public class ExplicitContentDetector {

	public static final String BASE_PATH_TO_INPUT_DATA =
			"data/input/lv/fifty/shades_labeled.txt";
	
	public static void main(String[] args) {
		
		WinConfig.setupEnv();
		
		//1) local spark conf and java spark context creation
		
		SparkConf cnf = new SparkConf().setMaster("local")
				.setAppName(ExplicitContentDetector.class.getName());
		
		SparkSession spark = SparkSession.builder()
				.config(cnf)
				.getOrCreate();
		
		//2) register our UDF to clean up text snippets
		
		spark.udf().register("textCleaner",
				PreprocessUtils.getUdfTextCleaner50Shades(),
				DataTypes.StringType
		);
		
		//3) load CSV file as DataFrame
		
		Dataset<Row> rawData = spark.read()
				.option("delimiter","\t")
				.csv(BASE_PATH_TO_INPUT_DATA);
		
		//4) transform the data frame with a little help
		//from the previously registered UDF
		
		Dataset<Row> snippetsCleaned = rawData
				.selectExpr("_c0 as passage",
						"textCleaner(_c1) as text",
						"_c2 as class");
		
		snippetsCleaned.show();
		
		//5) split data into training (75%) vs. test set (25%)
		
		Dataset<Row>[] splits = snippetsCleaned
				.randomSplit(new double[]{0.75,0.25},1234L);
		Dataset<Row> trainingSet = splits[0].cache();
		Dataset<Row> testSet= splits[1].cache();
		
		//6) build a pipeline with the following 4 stages
		//a) indexer -> b) tokenize -> c) hashing term frequency -> d) naive bayes
		
		StringIndexer idx = new StringIndexer()
				.setInputCol("class")
				.setOutputCol("label");
		
		Tokenizer tok = new Tokenizer()
				.setInputCol("text")
				.setOutputCol("words");
		
		HashingTF htf = new HashingTF()
				.setNumFeatures(4096)
				.setInputCol("words")
				.setOutputCol("features");
		
		NaiveBayes nb = new NaiveBayes()
				.setSmoothing(0.1)
				.setFeaturesCol("features")
				.setLabelCol("label");
		
		//7) configure the ML pipeline and use
		//trainingSet for model fitting
		
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] {idx, tok, htf, nb});
		
		PipelineModel pmodel = pipeline.fit(trainingSet);
		
		//8) apply trained model on test set and compute
		// the corresponding confusion matrix
		// evaluate model's metrics
		// Confusion matrix
		// precision
		// total accuracy
		
		Dataset<Row> predictionsTestData = pmodel.transform(testSet);
		//evaluate model's metrics
		MulticlassMetrics metrics =
				new MulticlassMetrics(
						predictionsTestData.select("prediction","label")
				);
		
		Matrix confusion= metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n"+ confusion);
		System.out.println("Precision MAYBE = "+ metrics.precision(0.0));
		System.out.println("Precision NO = "+ metrics.precision(1.0));
		System.out.println("Precision YES = "+ metrics.precision(2.0));
		System.out.println("Accuracy = "+ metrics.accuracy());
		
	}


}
