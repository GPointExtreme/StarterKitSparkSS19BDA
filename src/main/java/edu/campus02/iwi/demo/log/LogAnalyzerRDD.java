package edu.campus02.iwi.demo.log;

import java.io.Serializable;
import java.util.Comparator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;

import edu.campus02.iwi.demo.sql.LogAnalyzerDSL;
import edu.campus02.iwi.spark.env.WinConfig;
import scala.Tuple2;

/**
 * The LogAnalyzer takes in an apache access log file and computes some
 * statistics on them. The following statistics will be computed: - The average,
 * min, and max content size of responses returned from the server. - A count of
 * response code's returned. - All IPAddresses that have accessed this server
 * more than N times. - The top endpoints requested by count.
 */
public class LogAnalyzerRDD {

	private static Function2<Long, Long, Long> SUM_REDUCER = (a, b) -> a + b;

	@SuppressWarnings("serial")
	private static class ValueComparator<K, V> implements
			Comparator<Tuple2<K, V>>, Serializable {
		private Comparator<V> comparator;

		public ValueComparator(Comparator<V> comparator) {
			this.comparator = comparator;
		}

		@Override
		public int compare(Tuple2<K, V> o1, Tuple2<K, V> o2) {
			return comparator.compare(o1._2(), o2._2());
		}
	}

	public static void main(String[] args) {

		WinConfig.setupEnv();
		
		if (args.length != 1) {
			System.err.println("usage: program <path_to_apache_log_file>");
			System.exit(-1);
		}

		SparkConf cnf = new SparkConf().setMaster("local")
				.setAppName(LogAnalyzerDSL.class.getName());
		
		SparkSession spark = SparkSession.builder()
							     .config(cnf)
							     .getOrCreate();
		
		// Load the text file into Spark.
		JavaRDD<String> logLines = spark.read().text(args[0])
									.as(Encoders.STRING()).toJavaRDD();

		// Convert the text log lines to ApacheAccessLog objects and cache them
		// since multiple transformations and actions will be called on that
		// data.
		JavaRDD<ApacheAccessLog> accessLogs = logLines.map(
				ApacheAccessLog::parseFromLogLine).filter(al -> al != null).cache();

		// Calculate statistics based on the content size.
		// Note how the contentSizes are cached as well since multiple actions
		// are called on that RDD.
		JavaRDD<Long> contentSizes = accessLogs.map(
				ApacheAccessLog::getContentSize).cache();

		System.out.println(String.format(
				"Content Size Avg: %s, Min: %s, Max: %s",
				contentSizes.reduce(SUM_REDUCER) / contentSizes.count(),
				contentSizes.min(Comparator.naturalOrder()),
				contentSizes.max(Comparator.naturalOrder())));

		// Compute Response Code to Count.
		List<Tuple2<Integer, Long>> responseCodeToCount = accessLogs
				.mapToPair(log -> new Tuple2<>(log.getResponseCode(), 1L))
				.reduceByKey(SUM_REDUCER).take(100);
		
		System.out.println(String.format("Response code counts: %s",
				responseCodeToCount));

		// Any IPAddress that has accessed the server more than 10 times.
		List<String> ipAddresses = accessLogs
						.mapToPair(log -> new Tuple2<>(log.getIpAddress(), 1L))
						.reduceByKey(SUM_REDUCER)
						.filter(t -> t._2 > 10)
						.map(t -> t._1)
						.take(100);

		System.out.println(String.format("Any 100 IPAddresses > 10x access: %s",
							ipAddresses));

		// Top Endpoints
		List<Tuple2<String, Long>> topEndpoints = accessLogs
				.mapToPair(log -> new Tuple2<>(log.getEndpoint(), 1L))
				.reduceByKey(SUM_REDUCER)
				.top(10,new ValueComparator<>(Comparator.<Long>naturalOrder()));

		System.out.println(String.format("Top Endpoints: %s", topEndpoints));

		// Stop the Spark Context before exiting.
		spark.close();
	}
}
