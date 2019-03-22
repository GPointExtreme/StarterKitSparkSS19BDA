package lv.classification.nb.spam;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import edu.campus02.iwi.spark.env.WinConfig;
import spark.exercise.utils.PreprocessUtils;

public class SpamDetectorBasic {

	public static final String BASE_PATH_TO_INPUT_DATA =
			"data/input/lv/emails/CSDMC2010_SPAM/";

	public static void main(String[] args) {

		WinConfig.setupEnv();

		//1) local spark conf and java spark context creation
		
		SparkConf cnf = new SparkConf().setMaster("local")
				.setAppName(SpamDetectorBasic.class.getName());
		
		SparkSession spark = SparkSession.builder()
				.config(cnf)
				.getOrCreate();
		
		//2) register our UDF to read contents of a single mail msg
		
		spark.udf().register("readMailMsg",
				PreprocessUtils.getUdfMailMsgReader(
						BASE_PATH_TO_INPUT_DATA+"TRAINING/"),
							DataTypes.StringType);
		
		//3) load CSV file containing files and lables as DataFrame
		
		Dataset<Row> labelsInfo = spark.read()
				.option("delimiter"," ")
				.csv(BASE_PATH_TO_INPUT_DATA+"SPAMTrain.label");
		
		//4) transform the data frame with a little help from
		// the previously registered UDF
		
		Dataset<Row> trainMails = labelsInfo
				.selectExpr("CAST(_c0 as double) as label",
						"_c1 as file",
						"readMailMsg(_c1) as msg")
				.filter("msg != ''");
		
		trainMails.show();
		
		//5) split data into training (75%) vs. test set (25%)
		
		Dataset<Row>[] splits = trainMails
				.randomSplit(new double[]{0.75,0.25},1234L);
		
		Dataset<Row> trainingSet = splits[0].cache();
		Dataset<Row> testSet = splits[1].cache();
		
		//6) build a pipeline with the following 3 stages
		//a) tokenize -> b) hashing term frequency -> c) naive bayes
		
		Tokenizer tok = new Tokenizer()
				.setInputCol("msg")
				.setOutputCol("words");
		
		HashingTF htf = new HashingTF()
				.setNumFeatures(2048)
				.setInputCol("words")
				.setOutputCol("features");
		
		NaiveBayes nb = new NaiveBayes()
				.setFeaturesCol("features")
				.setLabelCol("label");
		
		//7) configure the ML pipeline and use
		// trainingSet for model fitting
		
		Pipeline pipeline= new Pipeline().setStages(
				new PipelineStage[] {tok,htf,nb}
		);
		
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
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("Confusion matrix: \n"+ confusion);
		System.out.println("Precision SPAM = "+metrics.precision(0.0));
		System.out.println("Precision NO-SPAM = "+metrics.precision(1.0));
		System.out.println("Accuracy = "+ metrics.accuracy());
		
	}
}
