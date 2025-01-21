#!/usr/bin/env python
# coding: utf-8
# Import the required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Step 1: Initialize a Spark session
spark = SparkSession.builder \
    .appName("Spotify_Read_Files") \
    .getOrCreate()

# Album Dataset
spotify_album_df=spark.read.format("parquet").load("/user/spark/vinish/spotify_data_sqoop/")

# Add all artist columns into one single "all_artist" column
spotify_album_df=spotify_album_df.withColumn("all_artist", concat(
                                                                    coalesce(col("artists"), lit("")),
                                                                    coalesce(col("artist_0"), lit("")),
                                                                    coalesce(col("artist_1"), lit("")),
                                                                    coalesce(col("artist_2"), lit("")),
                                                                    coalesce(col("artist_3"), lit("")),
                                                                    coalesce(col("artist_4"), lit("")),
                                                                    coalesce(col("artist_5"), lit("")),
                                                                    coalesce(col("artist_6"), lit("")),
                                                                    coalesce(col("artist_7"), lit("")),
                                                                    coalesce(col("artist_8"), lit("")),
                                                                    coalesce(col("artist_9"), lit("")),
                                                                    coalesce(col("artist_10"), lit("")),
                                                                    coalesce(col("artist_11"), lit(""))))

# drop the unwanted columns
spotify_album_df = spotify_album_df.drop("artists","artist_0", "artist_1", "artist_2","artist_3",
                                                        "artist_4","artist_5","artist_6","artist_7","artist_8",
                                                        "artist_9","artist_10","artist_11","duration_ms")

# Splitting release_date column into date only.
# Casting the datatype of release_date column from string into date.
spotify_album_df=spotify_album_df.withColumn('release_date',substring('release_date',1,10))\
                        .withColumn('release_date',col('release_date').cast('date'))

# Transformation on data
# Sorting the track_name,album_name,label,all_artist by album_popularity
spotify_album_df = spotify_album_df.withColumn("album_popularity",col("album_popularity").cast("integer"))

# Extract the year from the release_date column
spotify_album_df = spotify_album_df.withColumn('release_year', year(col('release_date')))

spotify_album_df.repartition(1).write.mode("overwrite").option("header", "true").csv("/user/spark/vinish/spotify_data_sqoop/transformed_album_dataset")


# Artist Dataset
spotify_artist_df=spark.read.format("parquet").load("/user/spark/vinish/spotify_data_sqoop/")

# drop the unwanted columns
spotify_artist_df = spotify_artist_df.drop("genre_0","genre_1", "genre_2", "genre_3","genre_4",
                                                        "genre_5","genre_6")

# Change the datatype of followers column from string to integer.
spotify_artist_df=spotify_artist_df.withColumn("followers",col("followers").cast("int"))

spotify_album_df.repartition(1).write.mode("overwrite").option("header", "true").csv("/user/spark/vinish/spotify_data_sqoop/transformed_artist_dataset")



# Track Dataset
spotify_track_df=spark.read.format("parquet").load("/user/spark/vinish/spotify_data_sqoop/")

spotify_track_join = spotify_track_df.join(spotify_album_df, spotify_album_df.track_id==spotify_track_df.id,"inner")


spotify_track_join = spotify_track_join.withColumn("release_year", year("release_date"))\
                                            .groupBy("explicit","release_year").agg(count(col("explicit")).alias("explicit_count"))\
                                            .orderBy(col("release_year"))

# Create a new variable to track the proportion of tracks with explicit content, among the total tracks per year.
spotify_track_join = spotify_track_join.withColumn("release_year", year("release_date"))\
                                        .filter(col("release_year").isNotNull())\
                                        .groupBy("release_year","explicit").agg(sum(col("total_tracks")).alias("total_tracks")
                                                                                ,count(col("explicit")).alias("explicit_count"))\
                                        .orderBy(col("release_year"))

# Add a new column for the proportion of explicit tracks
spotify_track_join = spotify_track_join.withColumn("explicit_proportion",
    round(when((col("total_tracks")>0) & (col("explicit")==1), col("explicit_count")/col("total_tracks"))\
    .otherwise((col("total_tracks")-col("explicit_count"))/col("total_tracks")),2))

spotify_album_df.repartition(1).write.mode("overwrite").option("header", "true").csv("/user/spark/vinish/spotify_data_sqoop/transformed_tracks_dataset")




# Feature Dataset
spotify_feature_df=spark.read.format("parquet").load("/user/spark/vinish/spotify_data_sqoop/")

# drop the unwanted columns
spotify_feature_df = spotify_feature_df.drop("type","uri", "track_href", "analysis_url")

spotify_album_df.repartition(1).write.mode("overwrite").option("header", "true").csv("/user/spark/vinish/spotify_data_sqoop/transformed_features_dataset")

