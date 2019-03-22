package edu.campus02.iwi.demo.sql;

import static org.apache.spark.sql.functions.*;

import java.util.List;
import java.util.Objects;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import edu.campus02.iwi.demo.log.ApacheAccessLog;
import edu.campus02.iwi.spark.env.WinConfig;

/**
 * LogAnalyzerSQL shows how to use SQL syntax with Spark.
 *
 */
public class LogAnalyzerDSL {

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
				
		Dataset<ApacheAccessLog> accessLogs = spark.read().text(args[0])
				.as(Encoders.STRING())
				.map(ApacheAccessLog::parseFromLogLine,Encoders.bean(ApacheAccessLog.class))
				.filter(Objects::nonNull);
		
		accessLogs.cache();
		
		// Show a few sample records and the schema
		accessLogs.show(15);
		accessLogs.printSchema();
		
		// Simple grouping by HTTP method
		accessLogs.groupBy("method").count().show();
		
		// Calculate top 25 requested resources
		accessLogs.groupBy("endpoint").count()
						.orderBy(col("count").desc())
							.show(25,false);
		
		// All IP addresses that have accessed the server more than N times.
		List<Row> ips = accessLogs.groupBy("ipAddress").count()
							.filter(col("count").gt(250))
								.collectAsList();
		System.out.println(ips);
		
		// aggregate functions based on contentSize of requested resources		
		accessLogs.agg(avg("contentSize"),max("contentSize"),
						min("contentSize"),sum("contentSize")).show();
		
		spark.stop();
	}
}