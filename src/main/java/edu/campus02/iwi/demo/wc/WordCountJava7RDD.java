package edu.campus02.iwi.demo.wc;

import java.util.Arrays;
import java.util.Iterator;
import java.util.UUID;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import edu.campus02.iwi.spark.env.WinConfig;
import scala.Tuple2;

public class WordCountJava7RDD {

	@SuppressWarnings("serial")
	public static void main(String[] args) {

		WinConfig.setupEnv();
		
		if(args.length != 2) {
			System.err.println("usage: program <input_dir> <output_dir>");
			System.exit(-1);
		}
		
		// The first thing a Spark program must do is to create a
		// JavaSparkContext object, which tells Spark how to access
		// a cluster. To create a SparkContext you first need to build
		// a SparkConf object that contains information about your
		// application.

		// The appName parameter is a name for your application to show on the
		// cluster UI. Master is a Spark, Mesos or YARN cluster URL, or the
		// special string "local" to run in local mode. In practice, when running
		// on a cluster, you will not want to hardcode master in the program, but
		// rather launch the application with spark-submit and receive it there.
		SparkConf cnf = new SparkConf().setMaster("local[1]")
				.setAppName(WordCountJava7RDD.class.getName());				

		JavaSparkContext jsc = new JavaSparkContext(cnf);

		JavaRDD<String> files = jsc.textFile(args[0]+"/*.txt");

		JavaRDD<String> words = files.flatMap(
					new FlatMapFunction<String, String>() {
						@Override
						public Iterator<String> call(String line) throws Exception {
							//typically you would also normalize
							//and apply stop word filtering as well...
							return Arrays.asList(line.split("\\s+")).iterator();
						}
				});
				

		JavaPairRDD<String, Integer> counts = words.mapToPair(
					new PairFunction<String, String, Integer>() {
						@Override
						public Tuple2<String, Integer> call(String w)
								throws Exception {
							return new Tuple2<String, Integer>(w, 1);
						}
					}).reduceByKey(
						new Function2<Integer, Integer, Integer>() {
							@Override
							public Integer call(Integer i, Integer j) throws Exception {
								return i + j;
							}
					});
		
		//saving our resulting JavaPairRDD with word and total count to local filesystem
		String uuid = UUID.randomUUID().toString();
		String destination = args[1]+"/run/"+uuid+"/txt/";
		System.out.println("saving result to "+destination);
		counts.saveAsTextFile(destination);
		
		// It's good practice to explicitly stop/close the context before
		// your program finishes :)
		jsc.close();		
		
	}

}
