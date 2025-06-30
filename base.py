#!/usr/bin/env python3
# -------- base.py  --------------------------------------------------
import math, time, argparse
from collections import defaultdict
from operator import add

from pyspark.sql import SparkSession, Row, Window
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, BucketedRandomProjectionLSH
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.storagelevel import StorageLevel

# --------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data",   required=True,  help="CSV file inside container, e.g. /app/outlier_dataset_500_50.csv")
ap.add_argument("--master", default="spark://spark-master:7077")
ap.add_argument("--p",      type=int, default=2)
ap.add_argument("--m_max",  type=int, default=5)
ap.add_argument("--k",      type=int, default=10)
args = ap.parse_args()

# --------------------------------------------------------------------
# Spark session (one per container run)
# --------------------------------------------------------------------
spark = (SparkSession.builder
         .appName("LOF_Benchmark")
         .master(args.master)          # cluster URL comes from CLI
         .getOrCreate())

t0 = time.perf_counter()

# --------------------------------------------------------------------
# 1) load & normalise
# --------------------------------------------------------------------
df = spark.read.csv(args.data, header=True, inferSchema=True).cache()
cols = df.columns
label_col   = cols[-1]
feature_cols= cols[:-1]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
df_vec = assembler.transform(df).cache()

scaler = MinMaxScaler(inputCol="features_vec", outputCol="scaled_vec")
df_scaled = scaler.fit(df_vec).transform(df_vec).cache()

df_scaled = df_scaled.withColumn("scaled_arr", vector_to_array("scaled_vec"))
df_norm = df_scaled.select(
    *[F.col("scaled_arr")[i].alias(feature_cols[i]) for i in range(len(feature_cols))],
    F.col(label_col)
).cache()

# --------------------------------------------------------------------
# 2) DDR (binning → density grid)
# --------------------------------------------------------------------
width = 1.0 / args.p
def to_binned_row(row):
    bins = [min(max(int(math.ceil(row[f] / width)), 1), args.p) for f in feature_cols]
    rec  = {f"{feature_cols[i]}_bin": bins[i] for i in range(len(feature_cols))}
    rec["cube"] = bins
    return Row(**rec)

schema = StructType(
    [StructField(f"{c}_bin", IntegerType(), False) for c in feature_cols] +
    [StructField("cube", ArrayType(IntegerType(), containsNull=False), False)]
)
df_binned = spark.createDataFrame(
    df_norm.select(*feature_cols).rdd.map(to_binned_row), schema
).cache()

cube_count = df_binned.rdd.map(lambda r: (tuple(r["cube"]), 1)).reduceByKey(add)
dg = cube_count.map(lambda x: (list(x[0]), x[1])).toDF(["cube","density"]) \
               .cache()

# --------------------------------------------------------------------
# 3) mRMRD feature selection (same as before)
# --------------------------------------------------------------------
bin_cols = [f"{c}_bin" for c in feature_cols]
df_dens = df_binned.join(dg, on="cube").select(*bin_cols, "density") \
                   .cache()
N = df_dens.count()

def rel_map(row):
    y = row["density"]
    for f in feature_cols:
        yield ((f, row[f"{f}_bin"], y), 1)

cont_rel = df_dens.rdd.flatMap(rel_map).reduceByKey(add).collect()
n_xy, n_x, n_y = defaultdict(int), defaultdict(int), defaultdict(int)
for (f,b,y),cnt in cont_rel:
    n_xy[(f,b,y)] = cnt
    n_x[(f,b)]   += cnt
    n_y[y]       += cnt

mi_fd = {f: sum((cnt/N)*math.log((cnt/N)/((n_x[(f,b)]/N)*(n_y[y]/N)))
           for ((ff,b,y),cnt) in n_xy.items() if ff==f)
         for f in feature_cols}
Th = sum(mi_fd.values()) / len(mi_fd)

selected, candidates, mi_red = [], feature_cols.copy(), {}
while candidates and len(selected) < args.m_max:
    if not selected:
        best = max(candidates, key=lambda x: mi_fd[x])
    else:
        s = selected[-1]
        def red_map(row):
            bs = row[f"{s}_bin"]
            for j in candidates:
                yield ((j, row[f"{j}_bin"], bs), 1)
        cont_rs = df_binned.rdd.flatMap(red_map).reduceByKey(add).collect()
        n_xy_rs,n_x_rs,n_s_rs = defaultdict(int),defaultdict(int),defaultdict(int)
        for (j,bj,bs),cnt in cont_rs:
            n_xy_rs[(j,bj,bs)] = cnt
            n_x_rs[(j,bj)]    += cnt
            n_s_rs[bs]        += cnt
        for j in candidates:
            if (j,s) not in mi_red:
                I_js = sum((cnt/N)*math.log((cnt/N)/((n_x_rs[(j,bj)]/N)*(n_s_rs[bs]/N)))
                           for ((jj,bj,bs),cnt) in n_xy_rs.items() if jj==j)
                mi_red[(j,s)] = mi_red[(s,j)] = I_js
        best, best_score = None, float("-inf")
        for j in candidates:
            rel     = mi_fd[j]
            red_avg = sum(mi_red[(j,t)] for t in selected)/len(selected)
            score   = rel - red_avg
            if score > best_score:
                best, best_score = j, score
    if mi_fd[best] < Th:
        break
    selected.append(best)
    candidates.remove(best)

# --------------------------------------------------------------------
# 4) LOF
# --------------------------------------------------------------------
df_proj = df_norm.select(*selected, label_col) \
                 .withColumn("id", F.monotonically_increasing_id()).cache()

assembler2 = VectorAssembler(inputCols=selected, outputCol="features_vec")
df_vec2 = assembler2.transform(df_proj).select("id","features_vec").cache()

lsh = BucketedRandomProjectionLSH(
        inputCol="features_vec", outputCol="hashes",
        bucketLength=math.sqrt(len(selected))/2)
model = lsh.fit(df_vec2)

max_dist = math.sqrt(len(selected))
pairs = model.approxSimilarityJoin(df_vec2, df_vec2, max_dist,"dist") \
            .select(F.col("datasetA.id").alias("pid"),
                    F.col("datasetB.id").alias("oid"),
                    "dist")
pairs = pairs.filter("pid < oid").unionByName(
            pairs.selectExpr("oid as pid","pid as oid","dist"))

w = Window.partitionBy("pid").orderBy("dist")
knn = pairs.withColumn("rn", F.row_number().over(w)) \
           .filter(F.col("rn")<=args.k) \
           .select("pid","oid","dist").cache()

kdist = knn.groupBy("oid").agg(F.max("dist").alias("kdist")).cache()
rd = knn.join(kdist, on="oid") \
        .withColumn("reach_dist",F.greatest("dist","kdist")) \
        .cache()
lrd = rd.groupBy("pid").agg((F.lit(args.k)/F.sum("reach_dist")).alias("lrd")) \
        .cache()
lof = rd.join(lrd.select(F.col("pid").alias("oid"),"lrd"), on="oid") \
        .groupBy("pid").agg(F.avg("lrd").alias("avg_lrd_o")) \
        .join(lrd,on="pid") \
        .withColumn("LOF",F.col("avg_lrd_o")/F.col("lrd")) \
        .cache()
        
elapsed = time.perf_counter() - t0
# --------------------------------------------------------------------
# 5) AUC
# --------------------------------------------------------------------
score_label = (df_proj.join(lof.withColumnRenamed("pid","id").select("id","LOF"),"id")
               .select("LOF", label_col)
               .rdd.map(lambda r: (float(r[0]), float(r[1]))))
auc = BinaryClassificationMetrics(score_label).areaUnderROC

print(f"Time={elapsed:.1f}s  AUC={auc:.4f}")
print("Selected subspace:", selected)

spark.stop()
