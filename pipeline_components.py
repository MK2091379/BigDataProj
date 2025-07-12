
import math
from typing import List, Tuple
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession, DataFrame, functions as F, Row
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark import StorageLevel
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# ─── 0) CSV loader + basic cleaning ────────────────────────────────────────────
def load_and_clean_csv(
    spark: SparkSession,
    path: str,
    *,
    id_col: str = "id",
    label_col: str = "target",
    drop_null_rows: bool = True,
) -> DataFrame:
    """
    Read CSV → drop duplicates → strip all-zero numeric cols → strip all-null cols
    → drop rows with any nulls → ensure an 'id' column → persist in memory.
    """
    df = spark.read.csv(path, header=True, inferSchema=True).dropDuplicates()

    # 1) drop numeric columns whose sum is zero
    numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "float", "double")]
    zero_sums = df.select([F.sum(F.col(c)).alias(c) for c in numeric_cols]).collect()[0].asDict()
    zero_cols = [c for c, s in zero_sums.items() if s == 0]

    # 2) drop columns that are entirely null
    total_rows = df.count()
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
    ]).collect()[0].asDict()
    null_only = [c for c, cnt in null_counts.items() if cnt == total_rows]

    drop_cols = list(set(zero_cols + null_only))
    if drop_cols:
        df = df.drop(*drop_cols)

    # 3) drop any rows containing nulls
    if drop_null_rows:
        df = df.na.drop()

    # 4) ensure an 'id' column
    if id_col not in df.columns:
        df = df.withColumn(id_col, F.monotonically_increasing_id())

    # 5) reorder so 'id' is first
    cols = [id_col] + [c for c in df.columns if c != id_col]
    return df.select(cols).persist(StorageLevel.MEMORY_ONLY)


# ─── 1) Feature assembly & scaling ─────────────────────────────────────────────
def assemble_vector(
    df: DataFrame, feature_cols: List[str], output_col: str = "features"
) -> DataFrame:
    return VectorAssembler(inputCols=feature_cols, outputCol=output_col).transform(df)


def minmax_scale(
    df: DataFrame,
    *,
    input_col: str = "features",
    output_col: str = "scaledFeatures",
) -> DataFrame:
    scaler = MinMaxScaler(inputCol=input_col, outputCol=output_col)
    return scaler.fit(df).transform(df)


def prepare_data(
    df: DataFrame, raw_cols: List[str]
) -> Tuple[DataFrame, List[str]]:
    df_vec = assemble_vector(df, raw_cols)
    df_scaled = minmax_scale(df_vec)
    return df_scaled.select("id", "scaledFeatures"), raw_cols


# ─── 2) Density‐representation & mutual‐information ────────────────────────────
def density_repr(
    df_scaled: DataFrame, feature_cols: List[str], p: int = 10
) -> Tuple[DataFrame, int]:
    """
    Bin each scaled feature into p bins, count densities of the resulting cubes.
    Returns density_df with columns [cube, density, g0, g1, …], and num_feats.
    """
    spark = df_scaled.sparkSession
    num_feats = len(feature_cols)

    # map each row to a cube string and count
    cube_counts = (
        df_scaled
        .select("scaledFeatures")
        .rdd
        .map(lambda r: tuple(int(min(x * p, p - 1)) for x in r.scaledFeatures))
        .map(lambda bins: ("_".join(map(str, bins)), 1))
        .reduceByKey(lambda a, b: a + b)
    )

    density_df = spark.createDataFrame(cube_counts, schema=["cube", "density"])
    for i in range(num_feats):
        density_df = density_df.withColumn(f"g{i}", F.split("cube", "_")[i].cast("int"))

    return density_df, num_feats


def compute_mi(
    num_feats: int, density_df: DataFrame
) -> Tuple[dict, dict]:
    """
    Compute relevance MI for each feature (mi_rel) and pairwise MI (mi_pair).
    """
    cols = [f"g{i}" for i in range(num_feats)]
    mr_rdd = density_df.select(cols + ["density"]).rdd.cache()
    N = mr_rdd.count()

    # feature–density joint counts
    fd = (
        mr_rdd.flatMap(
            lambda r: [((j, getattr(r, cols[j]), r.density), 1) for j in range(num_feats)]
        )
        .reduceByKey(lambda a, b: a + b)
    )
    feat_marg = fd.map(lambda kv: ((kv[0][0], kv[0][1]), kv[1])).reduceByKey(lambda a, b: a + b)
    dens_marg = fd.map(lambda kv: (kv[0][2], kv[1])).reduceByKey(lambda a, b: a + b)

    fd_list = fd.collect()
    feat_dict = dict(feat_marg.collect())
    dens_dict = dict(dens_marg.collect())

    # relevance MI
    mi_rel = {}
    for (j, vj, d), cnt in fd_list:
        p_jd = cnt / N
        mi_rel[j] = mi_rel.get(j, 0.0) + p_jd * math.log2(
            p_jd / ((feat_dict[(j, vj)] / N) * (dens_dict[d] / N))
        )

    # pairwise MI (redundancy)
    pair = (
        mr_rdd.flatMap(
            lambda r: [
                ((j, l, getattr(r, cols[j]), getattr(r, cols[l])), 1)
                for j in range(num_feats)
                for l in range(j + 1, num_feats)
            ]
        )
        .reduceByKey(lambda a, b: a + b)
    )
    from collections import defaultdict
    pair_dict = defaultdict(list)
    for (j, l, vj, vl), cnt in pair.collect():
        pair_dict[(j, l)].append(((vj, vl), cnt))

    mi_pair = {}
    for (j, l), lst in pair_dict.items():
        score = 0.0
        for (vj, vl), cnt in lst:
            p_jl = cnt / N
            score += p_jl * math.log2(
                p_jl / ((feat_dict[(j, vj)] / N) * (feat_dict[(l, vl)] / N))
            )
        mi_pair[(j, l)] = mi_pair[(l, j)] = score

    return mi_pair, mi_rel


def mrmd_select(
    num_feats: int, mi_pair: dict, mi_rel: dict, m: int
) -> List[int]:
    """
    Greedily select m features maximizing (relevance − redundancy).
    """
    selected, remaining = [], list(range(num_feats))
    while remaining and len(selected) < m:
        best, best_score = None, float("-inf")
        for c in remaining:
            redund = (
                0.0
                if not selected
                else sum(mi_pair.get((c, s), 0.0) for s in selected) / len(selected)
            )
            score = mi_rel.get(c, 0.0) - redund
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        remaining.remove(best)
    return selected


# ─── 3) Project into selected subspace ─────────────────────────────────────────
def project_subspace(
    df_scaled: DataFrame, selected: List[int]
) -> DataFrame:
    """
    From df_scaled(id, scaledFeatures), extract only the selected dimensions
    into a new DataFrame (id, [f0, f1, …]).
    """
    # expand vector → array
    arr_col = "feat_arr"
    df_arr = df_scaled.withColumn(arr_col, vector_to_array("scaledFeatures"))
    select_cols = ["id"] + [
        F.col(arr_col)[i].alias(f"f{idx}") for idx, i in enumerate(selected)
    ]
    return df_arr.select(*select_cols).persist(StorageLevel.MEMORY_ONLY)


# ─── 4) Compute LOF via sklearn ────────────────────────────────────────────────
def compute_lof(
    subspace_df: DataFrame, k: int = 50
) -> DataFrame:
    """
    Convert to pandas, run LocalOutlierFactor, return a Spark DataFrame with columns (id, lof_score).
    """
    pdf = subspace_df.toPandas().set_index("id")
    X = pdf.values
    lof = LocalOutlierFactor(n_neighbors=k, metric="euclidean")
    _ = lof.fit_predict(X)
    pdf["lof_score"] = -lof.negative_outlier_factor_
    spark = subspace_df.sparkSession
    return spark.createDataFrame(pdf.reset_index())


# ─── 5) Public pipeline runners ────────────────────────────────────────────────
def run_pipeline(
    df_scaled: DataFrame,
    feature_cols: List[str],
    *,
    p: int = 10,
    m: int = 5,
    k: int = 50,
) -> Tuple[DataFrame, List[int]]:
    """
    Given a scaled DataFrame and its feature columns, run:
      density_repr → compute_mi → mrmd_select → project_subspace → compute_lof
    Returns (lof_df, selected_feature_indices).
    """
    density_df, num_feats = density_repr(df_scaled, feature_cols, p)
    mi_pair, mi_rel = compute_mi(num_feats, density_df)
    selected = mrmd_select(num_feats, mi_pair, mi_rel, m)
    sub_df = project_subspace(df_scaled, selected)
    lof_df = compute_lof(sub_df, k)
    return lof_df, selected


def run_pipeline_from_csv(
    spark: SparkSession,
    csv_path: str,
    *,
    id_col: str = "id",
    label_col: str = "target",
    p: int = 10,
    m: int = 5,
    k: int = 50,
) -> Tuple[DataFrame, List[int], DataFrame]:
    """
    Read & clean CSV → prepare_data → run_pipeline.
    Returns (lof_df, selected_indices, cleaned_df).
    """
    df = load_and_clean_csv(spark, csv_path, id_col=id_col, label_col=label_col)
    feature_cols = [
        c
        for c, t in df.dtypes
        if c not in (id_col, label_col) and t in ("int", "bigint", "float", "double")
    ]
    df_scaled, _ = prepare_data(df, feature_cols)
    lof_df, selected = run_pipeline(df_scaled, feature_cols, p=p, m=m, k=k)
    return lof_df, selected, df,




def compute_pipeline_auc(
    spark: SparkSession,
    csv_path: str,
    minority_class,
    *,
    id_col: str = "id",
    label_col: str = "target",
    p: int = 10,
    m: int = 5,
    k: int = 50,
    pure_k: int = None
) -> Tuple[float, float]:

    # 1) Load & clean
    cleaned_df = load_and_clean_csv(spark, csv_path, id_col=id_col, label_col=label_col)
    # Prepare full scaled features for pure LOF
    feature_cols = [c for c, t in cleaned_df.dtypes if c not in (id_col, label_col) and t in ("int", "bigint", "float", "double")]
    df_vec = assemble_vector(cleaned_df, feature_cols, output_col="features")
    df_scaled = minmax_scale(df_vec, input_col="features", output_col="scaledFeatures")

    # 2) Compute pure LOF AUC
    pure_neighbors = pure_k or k
    pdf_pure = df_scaled.select(id_col, "scaledFeatures", label_col).toPandas()
    X_pure = np.vstack(pdf_pure["scaledFeatures"].values)
    y_pure = (pdf_pure[label_col] == minority_class).astype(int).values
    lof_pure = LocalOutlierFactor(n_neighbors=pure_neighbors, novelty=True)
    lof_pure.fit(X_pure)
    pdf_pure["lof_score_pure"] = -lof_pure.decision_function(X_pure)
    pure_auc = roc_auc_score(y_pure, pdf_pure["lof_score_pure"])

    # 3) Run pipeline LOF & compute its AUC
    lof_df, selected, _ = run_pipeline_from_csv(
        spark, csv_path,
        id_col=id_col, label_col=label_col,
        p=p, m=m, k=k
    )
    dfj = cleaned_df.join(lof_df, on=id_col)
    pdf_pipe = dfj.select(id_col, "lof_score", label_col).toPandas()
    y_true = (pdf_pipe[label_col] == minority_class).astype(int).values
    pipe_auc = roc_auc_score(y_true, pdf_pipe["lof_score"])

    return pipe_auc, pure_auc



def plot_pipeline_roc(
    spark: SparkSession,
    csv_path: str,
    minority_class,
    *,
    id_col: str = "id",
    label_col: str = "target",
    p: int = 10,
    m: int = 5,
    k: int = 50,
    lof_neighbors: int = 212,
):
    # 1) load, clean, and run pipeline
    lof_df, selected, cleaned_df = run_pipeline_from_csv(
        spark, csv_path, id_col=id_col, label_col=label_col, p=p, m=m, k=k
    )
    # 2) build eval_pdf with binary labels
    dfj = cleaned_df.join(lof_df, on=id_col)
    eval_pdf = dfj.select(id_col, "lof_score", label_col).toPandas()
    eval_pdf["binary_label"] = (eval_pdf[label_col] == minority_class).astype(int)

    # 3) run pure LOF on all features
    feature_cols = [c for c in cleaned_df.columns if c not in (id_col, label_col)]
    df_vec = VectorAssembler(inputCols=feature_cols, outputCol="vec").transform(cleaned_df)
    df_scaled = MinMaxScaler(inputCol="vec", outputCol="scaled").fit(df_vec).transform(df_vec)
    pdf_all = df_scaled.select(id_col, "scaled", label_col).toPandas()
    X_all = np.vstack(pdf_all["scaled"].values)
    y_all = (pdf_all[label_col] == minority_class).astype(int).values

    lof_pure = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
    lof_pure.fit(X_all)
    pdf_all["lof_score_pure"] = -lof_pure.decision_function(X_all)

    # 4) compute ROC data
    fpr_m, tpr_m, _ = roc_curve(eval_pdf["binary_label"], eval_pdf["lof_score"])
    fpr_p, tpr_p, _ = roc_curve(y_all, pdf_all["lof_score_pure"])
    auc_m = roc_auc_score(eval_pdf["binary_label"], eval_pdf["lof_score"])
    auc_p = roc_auc_score(y_all, pdf_all["lof_score_pure"])

    # 5) plot
    plt.figure(dpi=110, figsize=(6,5))
    plt.plot(fpr_m, tpr_m, label=f"mRMRD-LOF (AUC={auc_m:.2f})", linewidth=2)
    plt.plot(fpr_p, tpr_p, "--", label=f"Pure LOF (AUC={auc_p:.2f})", linewidth=2)
    plt.plot([0,1], [0,1], ":", label="Chance level")
    plt.xlim(0,1); plt.ylim(0,1.05)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC – Pipeline vs Pure LOF", fontsize=13)
    plt.grid(alpha=0.3); plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()

