package edu.campus02.iwi.demo.sql;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import edu.campus02.iwi.spark.env.WinConfig;

public class PurchaseStatsSQL {

	public static void main(String[] args) {

		WinConfig.setupEnv();
		
		if (args.length != 1) {
			System.err.println("usage: program <path_to_purchase_json>");
			System.exit(-1);
		}

		SparkConf cnf = new SparkConf().setMaster("local")
				.setAppName(LogAnalyzerDSL.class.getName());
		
		SparkSession spark = SparkSession.builder()
							     .config(cnf)
							     .getOrCreate();
		
		//create and  cache DataFrame of JSON file
		Dataset<Row> buyings = spark.read().json(args[0]).cache();

		buyings.createOrReplaceTempView("buyings");

		// Print the schema in a tree format
		buyings.printSchema();
		// Displays the content for N rows to stdout
		buyings.show(5);
		
		//grouping using good old SQL :)
		System.out.println("grouping with good old SQL");
		
		Dataset<Row> result = spark.sql(
					"SELECT productCategory,paymentType,buyingLocation,"
							+ " COUNT(*) AS numOrders,MIN(orderTotal) AS minOrder,"
							+ "	MAX(orderTotal) AS maxOrder FROM buyings"
							+ " GROUP BY productCategory,paymentType,buyingLocation");
		result.show(100);
		
		spark.stop();

	}

}
