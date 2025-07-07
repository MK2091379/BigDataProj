# pipeline.py

import argparse
import time
import math
import heapq
from collections import defaultdict

import pandas as pd
from sklearn.metrics import roc_auc_score

from pyspark.sql import SparkSession, Row, functions as F
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark import StorageLevel

def run_pipeline(csv_file: str, workers: int):
    # 1) Init Spark
    spark = (
        SparkSession.builder
        .appName("Dimension Reduction Pipeline")
        .master(f"local[{workers}]")
        .getOrCreate()
    )

    # 2) Load data
    pdf = pd.read_csv(csv_file)
    label_col = pdf.columns[-1]
    df = spark.createDataFrame(pdf)
    feature_cols = [c for c in df.columns if c != label_col]

    # 3) Assemble + MinMax scale
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    assembled_df = assembler.transform(df)
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scaler_model = scaler.fit(assembled_df)
    df_scaled = scaler_model.transform(assembled_df)

    # 4) Density-based representation
    p = 10
    num_feats = len(feature_cols)
    cube_counts = (
        df_scaled.select("scaledFeatures").rdd
        .map(lambda row: tuple(int(min(x * p, p - 1)) for x in row.scaledFeatures))
        .map(lambda bins: ("_".join(map(str, bins)), 1))
        .reduceByKey(lambda a, b: a + b)
    )
    density_df = spark.createDataFrame(cube_counts.map(lambda kv: Row(cube_id=kv[0], density=kv[1])))
    for i in range(num_feats):
        density_df = density_df.withColumn(f"g{i}", F.split(F.col("cube_id"), "_")[i].cast("int"))

    # 5) MI relevance I(gj; density)
    col_names = [f"g{i}" for i in range(num_feats)]
    mr_rdd = density_df.select(col_names + ["density"]).rdd.cache()
    N_total = mr_rdd.count()

    fd_counts = (
        mr_rdd.flatMap(lambda row: [((j, getattr(row, col_names[j]), row.density), 1)
                                     for j in range(num_feats)])
              .reduceByKey(lambda a, b: a + b)
    )
    feat_marg = fd_counts.map(lambda kv: ((kv[0][0], kv[0][1]), kv[1])).reduceByKey(lambda a, b: a + b)
    dens_marg = fd_counts.map(lambda kv: (kv[0][2], kv[1])).reduceByKey(lambda a, b: a + b)

    fd_list = fd_counts.collect()
    feat_dict = dict(feat_marg.collect())
    dens_dict = dict(dens_marg.collect())

    mi_relevance = {}
    for (j, gval, dc), cnt in fd_list:
        p_joint = cnt / N_total
        p_g = feat_dict[(j, gval)] / N_total
        p_d = dens_dict[dc] / N_total
        mi_relevance[j] = mi_relevance.get(j, 0.0) + p_joint * math.log2(p_joint / (p_g * p_d))

    # 6) MI redundancy I(gj; gl)
    pair_counts = (
        mr_rdd.flatMap(lambda row: [((j, l, getattr(row, col_names[j]), getattr(row, col_names[l])), 1)
                                     for j in range(num_feats) for l in range(j+1, num_feats)])
              .reduceByKey(lambda a, b: a + b)
    )
    pair_dict = defaultdict(list)
    for (j, l, vj, vl), c in pair_counts.collect():
        pair_dict[(j, l)].append(((vj, vl), c))

    mi_pair = {}
    for (j, l), items in pair_dict.items():
        score = 0.0
        for (vj, vl), cnt in items:
            p_joint = cnt / N_total
            p_j = feat_dict[(j, vj)] / N_total
            p_l = feat_dict[(l, vl)] / N_total
            score += p_joint * math.log2(p_joint / (p_j * p_l))
        mi_pair[(j, l)] = score
        mi_pair[(l, j)] = score

    # 7) mRMRD feature selection
    subspace_size = 10
    selected, remaining = [], list(range(num_feats))
    while remaining and len(selected) < subspace_size:
        best, best_score = None, float("-inf")
        for cand in remaining:
            redund = (sum(mi_pair.get((cand, s), 0.0) for s in selected) / len(selected)) if selected else 0.0
            score = mi_relevance[cand] - redund
            if score > best_score:
                best, best_score = cand, score
        selected.append(best)
        remaining.remove(best)

    print(f"Selected features (mRMRD order): {selected}")

    # 8) Project into selected subspace
    bc_idx = spark.sparkContext.broadcast(selected)
    norm_rdd = df_scaled.rdd.zipWithIndex().map(lambda t: (t[1], t[0].scaledFeatures))
    proj_rdd = norm_rdd.map(lambda kv: (kv[0], [kv[1][i] for i in bc_idx.value]))
    subspace_df = (
        proj_rdd
        .map(lambda kv: (kv[0], Vectors.dense(kv[1])))
        .toDF(["id", "subspaceFeatures"])
        .persist(StorageLevel.MEMORY_ONLY)
    )
    print(f"✓ Data mapping complete – projected into {len(selected)}-D subspace.")

    # 9) LOF scoring
    k = 50
    rdd = subspace_df.rdd.map(lambda r: (r.id, r.subspaceFeatures))
    pairs = (
        rdd.cartesian(rdd)
           .filter(lambda t: t[0][0] != t[1][0])
           .map(lambda t: (t[0][0], (Vectors.squared_distance(t[0][1], t[1][1]), t[1][0])))
    )
    def top_k(acc, x):
        if len(acc) < k:
            heapq.heappush(acc, (-x[0], x[1]))
        else:
            heapq.heappushpop(acc, (-x[0], x[1]))
        return acc

    knn = pairs.aggregateByKey([], top_k, lambda a, b: heapq.nsmallest(k, a + b, key=lambda t: -t[0]))
    kdist = knn.mapValues(lambda lst: max(-d for d, _ in lst)).collectAsMap()
    bc_kd = spark.sparkContext.broadcast(kdist)

    reach_rdd = knn.flatMap(lambda item: [(item[0], (max(-d2, bc_kd.value[j]), 1.0))
                                          for d2, j in item[1]])
    lrd = reach_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                   .mapValues(lambda s: s[1] / s[0])
    bc_lrd = spark.sparkContext.broadcast(lrd.collectAsMap())

    schema = StructType([StructField("id", LongType(), False),
                         StructField("lof", DoubleType(), False)])
    lof_rdd = knn.map(lambda item: (
        int(item[0]),
        float(sum(bc_lrd.value[j] for _, j in item[1]) / (k * bc_lrd.value[item[0]]))
    ))
    lof_df = spark.createDataFrame(lof_rdd, schema).persist(StorageLevel.MEMORY_ONLY)

    # 10) Join & AUC
    label_rdd = spark.sparkContext.parallelize(
        [(i, float(y)) for i, y in enumerate(pdf[label_col].tolist())]
    )
    label_df = spark.createDataFrame(
        label_rdd,
        StructType([StructField("id", LongType(), False),
                    StructField(label_col, DoubleType(), False)])
    )
    scores_pdf = lof_df.join(label_df, on="id").toPandas()
    auc = roc_auc_score(scores_pdf[label_col], scores_pdf["lof"])

    lof_df.orderBy("lof", ascending=False).show(5, truncate=False)
    spark.stop()

    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True, help="input CSV file")
    parser.add_argument("--workers", type=int, default=4, help="number of cores")
    args = parser.parse_args()

    start_time = time.time()
    auc = run_pipeline(args.csv, args.workers)
    elapsed   = time.time() - start_time

    print(f"\n>> Workers={args.workers}, Time={elapsed:.2f}s, AUC={auc:.4f}")
