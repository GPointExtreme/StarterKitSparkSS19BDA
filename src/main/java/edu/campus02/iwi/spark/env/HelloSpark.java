package edu.campus02.iwi.spark.env;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import lv.classification.nb.lenses.ContactLensClassifier;

public class HelloSpark {

	public static void main(String[] args) {

		WinConfig.setupEnv();

		SparkConf cnf = new SparkConf()
				.setMaster("local")
				.setAppName(ContactLensClassifier.class.getName());

		SparkSession spark = SparkSession.builder()
				.config(cnf).getOrCreate();

		Dataset<Row> raw = spark.read()
				.json("data/input/demo/hellospark.json");

		raw.printSchema();
		raw.show();
		
		spark.close();

	}

} //69
