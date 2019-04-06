package lv.recommender.alsmf;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import edu.campus02.iwi.spark.env.WinConfig;
import scala.collection.JavaConversions;
import spark.exercise.utils.MovieRating;
import spark.exercise.utils.MovieTitle;

import static org.apache.spark.sql.functions.*;

import java.util.Arrays;
import java.util.List;

public class MovieRecommendation {

	public static final String BASE_PATH_TO_INPUT_RATINGS = "data/input/lv/movies/ratings.dat";

	public static final String BASE_PATH_TO_INPUT_MOVIES = "data/input/lv/movies/movies.dat";

	public static void main(String[] args) {

		WinConfig.setupEnv();

		// 1) Spark Context Init
		
		SparkConf cnf = new SparkConf().setMaster("local[2]")
				.setAppName(MovieRecommendation.class.getName())
				.set("spark.executor.memory", "2g");
		
		SparkSession spark = SparkSession.builder()
				.config(cnf)
				.getOrCreate();
		
		// 2) read all ratings data
		
		Dataset<MovieRating> ratings = spark.read()
				.text(BASE_PATH_TO_INPUT_RATINGS)
				.as(Encoders.STRING())
				.map(MovieRating::parseRating,Encoders.bean(MovieRating.class))
				.cache();
		
		// 3) read all movie data
		
		Dataset<MovieTitle> movies = spark.read()
				.text(BASE_PATH_TO_INPUT_MOVIES)
				.as(Encoders.STRING())
				.map(MovieTitle::parseColonsFormat,Encoders.bean(MovieTitle.class))
				.cache();
		
		// 4) split ratings data into 90% training, 10% test
		
		Dataset<MovieRating>[] splits = ratings
				.randomSplit(new double[]{0.9,0.1},1234L);
		Dataset<MovieRating> trainingData = splits[0].cache();
		Dataset<MovieRating> testSet= splits[1].cache();

		
		// 5) Build the recommendation model using ALS on the training data
		
		ALS als = new ALS()
				.setRank(8)
				.setMaxIter(10)
				.setRegParam(0.1)
				.setUserCol("userId")
				.setItemCol("movieId")
				.setRatingCol("rating")
				.setColdStartStrategy("drop");
		
		ALSModel model = als.fit(trainingData);
		
		// 10) (skip this till the end) tune the model using the method for 9) see below 
		// then assign the model to the best based on hyper param tuning
		
		tuneALSModel(ratings);
		
		// 6) make predictions for testData
		
		model.setColdStartStrategy("drop");
		Dataset<Row> predictions= model.transform(testSet);

		// 7) Evaluate the model by computing the RMSE on the test data
		
		RegressionEvaluator evaluator = new RegressionEvaluator()
				.setMetricName("rmse")
				.setLabelCol("rating")
				.setPredictionCol("prediction");
		
		Double rmse = evaluator.evaluate(predictions);
		System.out.println("Root-mean-square error = " + rmse);
		
		// 8) make personalized recommendations by
		// getting top n movies for specific userId
		
		Integer[] userIds= {456,234,101,23};
		Dataset<Row> topMovies= getTopNMovieRecommendations(ratings,
				movies, model, userIds, 10);
		
		topMovies.show(false);
		
	}

	//9)
	private static ALSModel tuneALSModel(Dataset<MovieRating> training) {

		ALS als = new ALS()
				.setUserCol("userId")
				.setItemCol("movieId")
				.setRatingCol("rating")
				.setColdStartStrategy("drop");

		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { als });

		// 9) build a grid in order to test different parameter combinations
		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(als.rank(), new int[] { 4, 8 })
				.addGrid(als.maxIter(), new int[] { 10, 15 })
				.addGrid(als.regParam(), new double[] { 0.1, 0.05 })
				.build();

		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
						.setEstimator(pipeline)
						.setEvaluator(new RegressionEvaluator()
							.setMetricName("rmse")
							.setLabelCol("rating").setPredictionCol("prediction"))
						.setEstimatorParamMaps(paramGrid).setTrainRatio(0.9);

		// run train validation split to choose the best set of params
		TrainValidationSplitModel vmodel = trainValidationSplit.fit(training);

		// return the best found model
		PipelineModel best = (PipelineModel) vmodel.bestModel();
		return (ALSModel) best.stages()[0];
	}

	private static Dataset<Row> getTopNMovieRecommendations(Dataset<MovieRating> ratings,
			Dataset<MovieTitle> movies, ALSModel model, Object[] userIds, int topN) {
		
		Dataset<Row> predictItemsForUsers = ratings.select(model.getUserCol()).distinct()
				.filter(col(model.getUserCol()).isin(userIds));
		
		Dataset<Row> recommendedItems = model.recommendForUserSubset(predictItemsForUsers, topN);
		
		return recommendedItems.selectExpr("userId","explode(recommendations) AS rec")
							.selectExpr("userId", "rec.movieId AS movieId", "rec.rating AS rating")
							.join(movies,col("movieId").equalTo(col("id")),"inner")
							.selectExpr("userId","CONCAT(title,' (',year,') - ',rating) AS movieRating")
							.groupBy(col("userId"))
							.agg(collect_list("movieRating"));
	}

}
