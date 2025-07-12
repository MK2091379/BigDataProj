#%%
from pyspark.sql import SparkSession, functions as F, Row
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import math

#%%
spark = SparkSession.builder.appName("Dimension Reduction").master("local[*]").getOrCreate()

#%% md
# #### Import Musk version 2 dataset
#%%
df = (
    spark.read.option("header", True).option("inferSchema", True).csv("musk2.csv")
    .withColumn("id", F.monotonically_increasing_id()).cache()
)

label_col = "class"
feature_cols = [c for c in df.columns if c not in ("id", label_col)]

# labels from the SAME DataFrame (single ID lineage)
labels = df.select("id", (1 - F.col("class")).alias("label").cast("int")).cache()

#%% md
#  #### Parameters
# ##### K: LOF Neighbors
# ##### P: Bins Per Feature
# ##### M: Subspace Size
#%%
params = {
    'p': 2,
    'm': 2,
    'k': 54
}

#%% md
# #### Scale With Max_Min Normalization method
#%%
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
assembled_df = assembler.transform(df)

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
df_scaled = scaler.fit(assembled_df).transform(assembled_df).select("id", "scaledFeatures")

#%% md
# ### Density Based Representation
#%%

#%%
num_feats = len(feature_cols)
p = params['p']
cube_counts = (
    df_scaled.select("scaledFeatures").rdd
    .map(lambda row: tuple(int(min(x * p, p - 1)) for x in row.scaledFeatures))
    .map(lambda bins: ("_".join(map(str, bins)), 1))
    .reduceByKey(lambda a, b: a + b)
)

density_df = spark.createDataFrame(
    cube_counts.map(lambda kv: Row(cube_id=kv[0], density=kv[1]))
)

for i in range(num_feats):
    density_df = density_df.withColumn(f"g{i}", F.split("cube_id", "_")[i].cast("int"))

#%% md
# #### mRMD-Based Relevant Subspace Selection
#%%
# RDD format
col_names = [f"g{i}" for i in range(num_feats)]
mr_rdd    = density_df.select(col_names + ["density"]).rdd.cache()
N_total   = mr_rdd.count()

# calculate similarity
fd_counts = (
    mr_rdd.flatMap(
        lambda row: [((j, getattr(row, col_names[j]), row.density), 1) for j in range(num_feats)]
    ).reduceByKey(lambda a, b: a + b)
)

feat_marg = fd_counts.map(lambda kv: ((kv[0][0], kv[0][1]), kv[1])).reduceByKey(lambda a, b: a + b)

dens_marg = fd_counts.map(lambda kv: (kv[0][2], kv[1])).reduceByKey(lambda a, b: a + b)

fd_list      = fd_counts.collect()
feat_dict    = dict(feat_marg.collect())
dens_dict    = dict(dens_marg.collect())

mi_relevance = {}
for (j, gval, dc), cnt in fd_list:
    p_joint = cnt / N_total
    p_g     = feat_dict[(j, gval)] / N_total
    p_d     = dens_dict[dc] / N_total
    mi_relevance[j] = mi_relevance.get(j, 0.0) + p_joint * math.log2(p_joint / (p_g * p_d))

#%% md
# #### Compute I(gi,gj) And Redundancy
#%%
pair_counts = (
    mr_rdd.flatMap(
        lambda row: [(((j, l, getattr(row, col_names[j]), getattr(row, col_names[l]))), 1)
                      for j in range(num_feats) for l in range(j + 1, num_feats)]
    ).reduceByKey(lambda a, b: a + b)
)

# aggregate pair dictionaries ( (j , l) , ( (v1,v2),cnt))
from collections import defaultdict
pair_dict = defaultdict(list)
for ((j, l, vj, vl), c) in pair_counts.collect():
    pair_dict[(j, l)].append(((vj, vl), c))

# compute mutual information
mi_pair = {}
for (j, l), items in pair_dict.items():
    score = 0.0
    for (vj, vl), cnt in items:
        p_joint = cnt / N_total
        p_j     = feat_dict[(j, vj)] / N_total
        p_l     = feat_dict[(l, vl)] / N_total
        score  += p_joint * math.log2(p_joint / (p_j * p_l))
    mi_pair[(j, l)] = score
    mi_pair[(l, j)] = score
#%% md
# #### Greedy mRMD Selection
#%%
# select the number of desired subspace
subspace_size = params['m']
selected, remaining = [], list(range(num_feats))

while remaining and len(selected) < subspace_size:
    best, best_score = None, float("-inf")
    for cand in remaining:
        redund = 0.0
        if selected:
            redund = sum(mi_pair.get((cand, s), 0.0) for s in selected) / len(selected)
        score = mi_relevance[cand] - redund
        if score > best_score:
            best, best_score = cand, score
    selected.append(best)
    remaining.remove(best)

# see the selected features
print("\nSelected features (MRMD order):", [f"g{i}" for i in selected])
#%% md
# #### Stage 5: Data Mapping
#%%
indexes = [int(col) for col in selected]  # from mRMRD
df_subspace = (
    df_scaled.withColumn("feat_arr", vector_to_array("scaledFeatures"))
    .select("id", *[F.col("feat_arr")[i].alias(f"f{j}") for j,i in enumerate(indexes)])
    .cache()
)

df_subspace.show(5, truncate=False)

#%% md
# #### Stage 6: Compute LOF Scores
# #### First Find K Nearest Neighbors
#%%
# --- Cell 20 (updated) ---

pdf = df_subspace.toPandas().set_index("id")
X = pdf.values

lof = LocalOutlierFactor(n_neighbors=params['k'], metric="euclidean")
lof.fit_predict(X)
# invert sign so that higher => more anomalous
pdf["lof_score"] = -lof.negative_outlier_factor_

lof_df = spark.createDataFrame(pdf.reset_index())
df_final = df_scaled.join(lof_df.select("id", "lof_score"), on="id", how="inner")

df_final.select("id", "lof_score").orderBy(F.desc("lof_score")).show(10, truncate=False)

#%% md
# #### Compute Local Reachability Density
#%%
lof_df = spark.createDataFrame(
    pdf.reset_index()[["id", "lof_score"]]
)

df_final = (
    df_scaled
      .join(lof_df, on="id", how="inner")
      .orderBy(F.desc("lof_score"))
)

df_final.select("id", "lof_score").show(10, truncate=False)
#%% md
# 
#%%
# --- Cell 13 (fixed) ---

# join the manually computed LOF scores and true labels
df_scored = (
    df_final
      .join(labels, on="id", how="inner")
      # select the column you really populated!
      .select("lof_score", "label")
      .dropna()
)

# compute AUC using the correct score column
pdf_auc = df_scored.toPandas()
auc = roc_auc_score(pdf_auc["label"], pdf_auc["lof_score"])

print(f"✅ Final Correct ROC-AUC: {auc:.4f}")

#%% md
# #### Hyper-Tuning Parameters Using The Functions From Pipline_Components
#%%

#%%

#%% md
# #### Test With Different Number Of Workers
#%%

#%%

#%%
df.to_csv('results.csv', index=False)
df_copy = df
#%%
print("Spark version:", spark.version)
#%%
