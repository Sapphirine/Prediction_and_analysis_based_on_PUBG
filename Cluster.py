# Databricks notebook source
pubg = spark.table("pubg_player_statistics_csv")
display(pubg.select("*"))

# COMMAND ----------

g = ['tracker_id', 'solo_KillDeathRatio', 'solo_WinRatio', 'solo_TimeSurvived', 'solo_RoundsPlayed', 'solo_Wins', 'solo_WinTop10Ratio', 'solo_Top10s', 'solo_Top10Ratio', 'solo_Losses', 'solo_Rating', 'solo_BestRating', 'solo_DamagePg', 'solo_HeadshotKillsPg', 'solo_HealsPg', 'solo_KillsPg', 'solo_MoveDistancePg', 'solo_RevivesPg', 'solo_RoadKillsPg', 'solo_TeamKillsPg', 'solo_TimeSurvivedPg', 'solo_Top10sPg', 'solo_Kills', 'solo_Assists', 'solo_Suicides', 'solo_TeamKills', 'solo_HeadshotKills', 'solo_HeadshotKillRatio', 'solo_VehicleDestroys', 'solo_RoadKills', 'solo_DailyKills', 'solo_WeeklyKills', 'solo_RoundMostKills', 'solo_MaxKillStreaks', 'solo_WeaponAcquired', 'solo_Days', 'solo_LongestTimeSurvived', 'solo_MostSurvivalTime', 'solo_AvgSurvivalTime', 'solo_WinPoints', 'solo_WalkDistance', 'solo_RideDistance', 'solo_MoveDistance', 'solo_AvgWalkDistance', 'solo_AvgRideDistance', 'solo_LongestKill', 'solo_Heals', 'solo_Revives', 'solo_Boosts', 'solo_DamageDealt', 'solo_DBNOs', 'duo_KillDeathRatio', 'duo_WinRatio', 'duo_TimeSurvived', 'duo_RoundsPlayed', 'duo_Wins', 'duo_WinTop10Ratio', 'duo_Top10s', 'duo_Top10Ratio', 'duo_Losses', 'duo_Rating', 'duo_BestRating', 'duo_DamagePg', 'duo_HeadshotKillsPg', 'duo_HealsPg', 'duo_KillsPg', 'duo_MoveDistancePg', 'duo_RevivesPg', 'duo_RoadKillsPg', 'duo_TeamKillsPg', 'duo_TimeSurvivedPg', 'duo_Top10sPg', 'duo_Kills', 'duo_Assists', 'duo_Suicides', 'duo_TeamKills', 'duo_HeadshotKills', 'duo_HeadshotKillRatio', 'duo_VehicleDestroys', 'duo_RoadKills', 'duo_DailyKills', 'duo_WeeklyKills', 'duo_RoundMostKills', 'duo_MaxKillStreaks', 'duo_WeaponAcquired', 'duo_Days', 'duo_LongestTimeSurvived', 'duo_MostSurvivalTime', 'duo_AvgSurvivalTime', 'duo_WinPoints', 'duo_WalkDistance', 'duo_RideDistance', 'duo_MoveDistance', 'duo_AvgWalkDistance', 'duo_AvgRideDistance', 'duo_LongestKill', 'duo_Heals', 'duo_Revives', 'duo_Boosts', 'duo_DamageDealt', 'duo_DBNOs', 'squad_KillDeathRatio', 'squad_WinRatio', 'squad_TimeSurvived', 'squad_RoundsPlayed', 'squad_Wins', 'squad_WinTop10Ratio', 'squad_Top10s', 'squad_Top10Ratio', 'squad_Losses', 'squad_Rating', 'squad_BestRating', 'squad_DamagePg', 'squad_HeadshotKillsPg', 'squad_HealsPg', 'squad_KillsPg', 'squad_MoveDistancePg', 'squad_RevivesPg', 'squad_RoadKillsPg', 'squad_TeamKillsPg', 'squad_TimeSurvivedPg', 'squad_Top10sPg', 'squad_Kills', 'squad_Assists', 'squad_Suicides', 'squad_TeamKills', 'squad_HeadshotKills', 'squad_HeadshotKillRatio', 'squad_VehicleDestroys', 'squad_RoadKills', 'squad_DailyKills', 'squad_WeeklyKills', 'squad_RoundMostKills', 'squad_MaxKillStreaks', 'squad_WeaponAcquired', 'squad_Days', 'squad_LongestTimeSurvived', 'squad_MostSurvivalTime', 'squad_AvgSurvivalTime', 'squad_WinPoints', 'squad_WalkDistance', 'squad_RideDistance', 'squad_MoveDistance', 'squad_AvgWalkDistance', 'squad_AvgRideDistance', 'squad_LongestKill', 'squad_Heals', 'squad_Revives', 'squad_Boosts', 'squad_DamageDealt', 'squad_DBNOs']
for i in g:
  pubg = pubg.withColumn(i, pubg[i].cast("double"))


# COMMAND ----------

pubg.drop('player_name').collect()
pubg = pubg.drop('player_name')
display(pubg.select("*"))

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=['solo_KillDeathRatio', 'solo_WinRatio', 'solo_TimeSurvived', 'solo_RoundsPlayed', 'solo_Wins', 'solo_WinTop10Ratio', 'solo_Top10s', 'solo_Top10Ratio', 'solo_Losses', 'solo_Rating', 'solo_BestRating', 'solo_DamagePg', 'solo_HeadshotKillsPg', 'solo_HealsPg', 'solo_KillsPg', 'solo_MoveDistancePg', 'solo_RevivesPg', 'solo_RoadKillsPg', 'solo_TeamKillsPg', 'solo_TimeSurvivedPg', 'solo_Top10sPg', 'solo_Kills', 'solo_Assists', 'solo_Suicides', 'solo_TeamKills', 'solo_HeadshotKills', 'solo_HeadshotKillRatio', 'solo_VehicleDestroys', 'solo_RoadKills', 'solo_DailyKills', 'solo_WeeklyKills', 'solo_RoundMostKills', 'solo_MaxKillStreaks', 'solo_WeaponAcquired', 'solo_Days', 'solo_LongestTimeSurvived', 'solo_MostSurvivalTime', 'solo_AvgSurvivalTime', 'solo_WinPoints', 'solo_WalkDistance', 'solo_RideDistance', 'solo_MoveDistance', 'solo_AvgWalkDistance', 'solo_AvgRideDistance', 'solo_LongestKill', 'solo_Heals', 'solo_Revives', 'solo_Boosts', 'solo_DamageDealt', 'solo_DBNOs', 'duo_KillDeathRatio', 'duo_WinRatio', 'duo_TimeSurvived', 'duo_RoundsPlayed', 'duo_Wins', 'duo_WinTop10Ratio', 'duo_Top10s', 'duo_Top10Ratio', 'duo_Losses', 'duo_Rating', 'duo_BestRating', 'duo_DamagePg', 'duo_HeadshotKillsPg', 'duo_HealsPg', 'duo_KillsPg', 'duo_MoveDistancePg', 'duo_RevivesPg', 'duo_RoadKillsPg', 'duo_TeamKillsPg', 'duo_TimeSurvivedPg', 'duo_Top10sPg', 'duo_Kills', 'duo_Assists', 'duo_Suicides', 'duo_TeamKills', 'duo_HeadshotKills', 'duo_HeadshotKillRatio', 'duo_VehicleDestroys', 'duo_RoadKills', 'duo_DailyKills', 'duo_WeeklyKills', 'duo_RoundMostKills', 'duo_MaxKillStreaks', 'duo_WeaponAcquired', 'duo_Days', 'duo_LongestTimeSurvived', 'duo_MostSurvivalTime', 'duo_AvgSurvivalTime', 'duo_WinPoints', 'duo_WalkDistance', 'duo_RideDistance', 'duo_MoveDistance', 'duo_AvgWalkDistance', 'duo_AvgRideDistance', 'duo_LongestKill', 'duo_Heals', 'duo_Revives', 'duo_Boosts', 'duo_DamageDealt', 'duo_DBNOs', 'squad_KillDeathRatio', 'squad_WinRatio', 'squad_TimeSurvived', 'squad_RoundsPlayed', 'squad_Wins', 'squad_WinTop10Ratio', 'squad_Top10s', 'squad_Top10Ratio', 'squad_Losses', 'squad_Rating', 'squad_BestRating', 'squad_DamagePg', 'squad_HeadshotKillsPg', 'squad_HealsPg', 'squad_KillsPg', 'squad_MoveDistancePg', 'squad_RevivesPg', 'squad_RoadKillsPg', 'squad_TeamKillsPg', 'squad_TimeSurvivedPg', 'squad_Top10sPg', 'squad_Kills', 'squad_Assists', 'squad_Suicides', 'squad_TeamKills', 'squad_HeadshotKills', 'squad_HeadshotKillRatio', 'squad_VehicleDestroys', 'squad_RoadKills', 'squad_DailyKills', 'squad_WeeklyKills', 'squad_RoundMostKills', 'squad_MaxKillStreaks', 'squad_WeaponAcquired', 'squad_Days', 'squad_LongestTimeSurvived', 'squad_MostSurvivalTime', 'squad_AvgSurvivalTime', 'squad_WinPoints', 'squad_WalkDistance', 'squad_RideDistance', 'squad_MoveDistance', 'squad_AvgWalkDistance', 'squad_AvgRideDistance', 'squad_LongestKill', 'squad_Heals', 'squad_Revives', 'squad_Boosts', 'squad_DamageDealt', 'squad_DBNOs'],
    outputCol="features")
dataset = assembler.transform(pubg)
print(dataset.select("features", "tracker_id").first())


# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
    inputCols=['solo_KillDeathRatio', 'solo_WinRatio', 'solo_HeadshotKillsPg', 'solo_HealsPg', 'solo_KillsPg', 'solo_RoadKillsPg', 'solo_TeamKillsPg', 'solo_Top10sPg'],
    outputCol="features")
dataset = assembler.transform(pubg)
print(dataset.select("features", "tracker_id").first())

# COMMAND ----------

from pyspark.ml.clustering import KMeans
# Trains a k-means model.
kmeans = KMeans().setK(10).setSeed(1)
model = kmeans.fit(dataset)
# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

aaa = model.transform(dataset)
display(aaa.select('solo_Rating','duo_Rating','squad_Rating','prediction'))

# COMMAND ----------

display(model)

# COMMAND ----------


