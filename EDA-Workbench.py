# Run: streamlit run eda_workbench.py

import io
import json
import base64
import re
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import umap

from scipy.stats import boxcox, ttest_ind, ks_2samp, skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif

import yaml

# Optional: pandas-profiling
try:
    import pandas_profiling
    PPROF_AVAILABLE = True
except Exception:
    PPROF_AVAILABLE = False

# ----------------------
# Utilities
# ----------------------
def _download_link_bytes(data: bytes, filename: str, mime: str, label=None):
    b64 = base64.b64encode(data).decode()
    text = label or f"Download {filename}"
    return f'<a href="data:{mime};base64,{b64}" download="{filename}">{text}</a>'

def _infer_schema(df: pd.DataFrame):
    sch = {"numeric": [], "categorical": [], "datetime": [], "text": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            sch["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            sch["datetime"].append(col)
        elif pd.api.types.is_string_dtype(s):
            ratio = s.nunique(dropna=True) / max(len(s), 1)
            (sch["categorical"] if ratio < 0.5 else sch["text"]).append(col)
        else:
            sch["categorical"].append(col)
    return sch

def _sample_df(df: pd.DataFrame, max_rows=5000, seed=42):
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=seed)
    return df

def _vif_frame(df: pd.DataFrame, cols):
    X = df[cols].dropna()
    if len(X) < 10 or X.shape[1] < 2:
        return None
    vifs = [_vif(X.values, i) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": cols, "VIF": vifs})

def _ml_readiness_score(df: pd.DataFrame, schema, target=None):
    miss = df.isna().mean().mean()
    cols = df.shape[1]
    num = len(schema["numeric"])
    cat = len(schema["categorical"])
    score = 100
    if miss > 0.1: score -= 20
    if num + cat < cols: score -= 10
    if cols > 200: score -= 10
    if target is not None and target in df.columns:
        vc = df[target].value_counts(normalize=True, dropna=True)
        if len(vc) > 1 and (vc.min() < 0.1):
            score -= 10
    return max(0, int(score))

def _embedded_sample_csv():
    sample = pd.DataFrame({
        "age": [25, 30, 35, 40, np.nan, 28, 32, 29, 41, 36],
        "income": [50000, 60000, 55000, 65000, 62000, 47000, 59000, 61000, 72000, 68000],
        "gender": ["M", "F", "F", "M", "F", "F", "M", "M", "M", "F"],
        "city": ["Dhaka", "Chittagong", "Dhaka", "Khulna", "Sylhet", "Dhaka", "Rajshahi", "Barishal", "Dhaka", "Chittagong"],
        "joined": pd.date_range("2021-01-01", periods=10, freq="MS"),
        "target": [0,1,0,1,0,0,1,0,1,1]
    })
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue()

def _stats_summary(series_before, series_after):
    def s(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) == 0:
            return {"mean": None, "std": None, "skew": None, "kurtosis": None}
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "skew": float(skew(x)) if len(x) > 2 else 0.0,
            "kurtosis": float(kurtosis(x)) if len(x) > 3 else 0.0
        }
    sb, sa = s(series_before), s(series_after)
    return sb, sa

# ----------------------
# App State
# ----------------------
st.set_page_config(page_title="Interactive EDA & ML-safe Workbench", layout="wide")

if "raw_df" not in st.session_state: st.session_state.raw_df = None
if "df" not in st.session_state: st.session_state.df = None
if "schema" not in st.session_state: st.session_state.schema = {"numeric":[], "categorical":[], "datetime":[], "text":[]}
if "target" not in st.session_state: st.session_state.target = None
if "versions" not in st.session_state: st.session_state.versions = {}
if "pipeline" not in st.session_state: st.session_state.pipeline = []
if "undo_stack" not in st.session_state: st.session_state.undo_stack = []
if "redo_stack" not in st.session_state: st.session_state.redo_stack = []
if "selection_mask" not in st.session_state: st.session_state.selection_mask = None
if "mode" not in st.session_state: st.session_state.mode = "Exploratory"
if "splits" not in st.session_state: st.session_state.splits = {"train": None, "val": None, "test": None}
if "fitted_transformers" not in st.session_state: st.session_state.fitted_transformers = {}
if "last_explain" not in st.session_state: st.session_state.last_explain = None

# ----------------------
# Functions for state ops
# ----------------------
def load_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_parquet(file)
    st.session_state.raw_df = df.copy()
    st.session_state.df = df.copy()
    st.session_state.schema = _infer_schema(st.session_state.df)
    st.session_state.selection_mask = None
    st.session_state.pipeline = []
    st.session_state.undo_stack.clear()
    st.session_state.redo_stack.clear()
    st.session_state.splits = {"train": None, "val": None, "test": None}
    st.session_state.fitted_transformers = {}
    st.session_state.last_explain = None

def checkpoint():
    st.session_state.undo_stack.append(st.session_state.df.copy())
    st.session_state.redo_stack.clear()

def undo_last():
    if st.session_state.undo_stack:
        st.session_state.redo_stack.append(st.session_state.df.copy())
        st.session_state.df = st.session_state.undo_stack.pop()
        st.session_state.schema = _infer_schema(st.session_state.df)
        if st.session_state.pipeline:
            st.session_state.pipeline.pop()
        st.session_state.last_explain = None
        st.success("Undid last operation.")
    else:
        st.info("Nothing to undo.")

def redo_last():
    if st.session_state.redo_stack:
        st.session_state.undo_stack.append(st.session_state.df.copy())
        st.session_state.df = st.session_state.redo_stack.pop()
        st.session_state.schema = _infer_schema(st.session_state.df)
        st.success("Redid operation.")
    else:
        st.info("Nothing to redo.")

def log_op(op, cols, params):
    st.session_state.pipeline.append({"op": op, "columns": cols, "params": params})

def save_version(name):
    st.session_state.versions[name] = st.session_state.df.copy()

def load_version(name):
    st.session_state.df = st.session_state.versions[name].copy()
    st.session_state.schema = _infer_schema(st.session_state.df)

def set_dtype(col, dtype):
    df = st.session_state.df
    if dtype == "numeric":
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif dtype == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    elif dtype == "categorical":
        df[col] = df[col].astype("category")
    elif dtype == "text":
        df[col] = df[col].astype("string")
    st.session_state.df = df
    st.session_state.schema = _infer_schema(df)

def apply_pipeline(pipeline, df=None, strict=False):
    df = st.session_state.df if df is None else df
    for step in pipeline:
        op, cols, params = step.get("op"), step.get("columns", []), step.get("params", {})
        missing = [c for c in cols if c not in df.columns]
        if missing:
            if strict:
                st.error(f"Missing columns for {op}: {missing}")
                break
            else:
                st.info(f"Skipping {op}; missing columns: {missing}")
                continue
        try:
            if op == "impute":
                strat = params.get("strategy","mean")
                if strat in ["ffill","bfill"]:
                    for c in cols:
                        df[c] = df[c].ffill() if strat == "ffill" else df[c].bfill()
                elif strat == "knn":
                    knn = KNNImputer()
                    df[cols] = knn.fit_transform(df[cols])
                else:
                    imp = SimpleImputer(strategy=strat, fill_value=params.get("fill_value"))
                    df[cols] = imp.fit_transform(df[cols])
            elif op == "encode":
                method = params.get("method","onehot")
                if method == "onehot":
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                        drop=("first" if params.get("drop_first") else None))
                    enc = ohe.fit_transform(df[cols])
                    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(cols), index=df.index)
                    df = df.drop(columns=cols).join(enc_df)
                elif method == "ordinal":
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    df[cols] = enc.fit_transform(df[cols])
                elif method == "label":
                    for c in cols:
                        le = LabelEncoder()
                        df[c] = le.fit_transform(df[c].astype(str))
                elif method.startswith("target"):
                    target = params.get("target")
                    if target in df.columns:
                        for c in cols:
                            means = df.groupby(c)[target].mean()
                            df[c] = df[c].map(means)
            elif op == "rare_group":
                thr = float(params.get("threshold_pct", 1.0)) / 100.0
                for c in cols:
                    freq = df[c].value_counts(normalize=True)
                    rares = freq[freq < thr].index
                    df[c] = df[c].apply(lambda x: "Other" if x in rares else x)
            elif op == "scale":
                meth = params.get("method","standard")
                scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[meth]
                df[cols] = scaler.fit_transform(df[cols])
            elif op == "transform":
                meth = params.get("method","log")
                if meth == "log":
                    for c in cols: df[c] = np.log1p(df[c].clip(lower=0))
                elif meth == "boxcox":
                    for c in cols:
                        x = df[c].clip(lower=1e-6)
                        df[c], _ = boxcox(x)
                elif meth == "yeo-johnson":
                    pt = PowerTransformer(method="yeo-johnson")
                    df[cols] = pt.fit_transform(df[cols])
        except Exception as e:
            st.warning(f"Step '{op}' failed: {e}")
    return df

# ----------------------
# App header and sidebar
# ----------------------
st.title("🔬 Interactive EDA & ML-safe Preprocessing Workbench")

st.sidebar.header("⚙️ Mode & Upload")
st.session_state.mode = st.sidebar.radio("Mode", ["Exploratory", "Modeling-safe"])

file = st.sidebar.file_uploader("Upload CSV/Excel/Parquet", type=["csv","xlsx","xls","parquet"])
if file:
    load_file(file)

st.sidebar.markdown("---")
st.sidebar.markdown("📊 Need a dataset?")
sample_csv_bytes = _embedded_sample_csv()
st.sidebar.markdown(_download_link_bytes(sample_csv_bytes, "sample.csv", "text/csv", "📥 Download Sample CSV"), unsafe_allow_html=True)

if st.session_state.df is None:
    st.info("👆 Upload a dataset to begin your analysis.")
    st.stop()

# ----------------------
# Top summary
# ----------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("📊 Rows", st.session_state.df.shape[0])
with c2: st.metric("📋 Columns", st.session_state.df.shape[1])
with c3: st.metric("❌ Missing cells", int(st.session_state.df.isna().sum().sum()))
with c4: st.metric("💾 Versions", len(st.session_state.versions))
with c5: st.metric("✅ ML readiness", _ml_readiness_score(st.session_state.df, st.session_state.schema, st.session_state.target))

st.write("**Schema:**", st.session_state.schema)
st.dataframe(st.session_state.df.head(20), use_container_width=True)

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📝 Schema Editor", "🔍 Diagnostics", "📊 Visualization", "⚙️ Preprocessing",
    "🎯 Dimensionality", "📈 Correlation & Stats",
    "📜 History & Versions", "💾 Export", "✨ Validation & Explainability"
])

# ----------------------
# Tab 1: Schema editor
# ----------------------
with tab1:
    st.subheader("Schema Editor & Target Selection")
    col_schema = st.selectbox("Select Column", list(st.session_state.df.columns), key="schema_col_select")
    dtype = st.selectbox("Set Data Type", ["numeric","categorical","datetime","text"], key="schema_dtype_select")
    if st.button("Apply Data Type", key="apply_dtype_btn"):
        checkpoint()
        set_dtype(col_schema, dtype)
        log_op("set_dtype", [col_schema], {"dtype": dtype})
        st.success(f"✅ Set {col_schema} → {dtype}")
    
    target = st.selectbox("Target Column (optional)", ["None"] + list(st.session_state.df.columns), key="target_select")
    if target != "None":
        st.session_state.target = target
        st.info(f"🎯 Target set: {target}")
    
    st.write("**Cardinality (unique counts):**")
    st.dataframe(st.session_state.df.nunique().sort_values(ascending=False).to_frame("unique"))

    st.markdown("---")
    st.markdown("### 🔀 Train/Validation/Test Split (Modeling-safe)")
    test_size = st.slider("Test size (%)", 10, 50, 20, key="test_size_slider")
    val_size = st.slider("Validation size (%) of remaining", 0, 30, 0, key="val_size_slider")
    stratify = st.checkbox("Stratify by target (if classification)", value=False, key="stratify_check")
    split_btn = st.button("Perform Split", key="perform_split_btn")
    if split_btn:
        if st.session_state.target is None:
            st.warning("⚠️ Set a target column first.")
        else:
            y = st.session_state.df[st.session_state.target]
            X = st.session_state.df.drop(columns=[st.session_state.target])
            if stratify and y.nunique() > 1:
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)
            else:
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            if val_size > 0:
                val_ratio = val_size / 100.0
                if stratify and y_temp.nunique() > 1:
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), random_state=42, stratify=y_temp)
                else:
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_ratio), random_state=42)
            else:
                X_val, y_val = None, None
                X_test, y_test = X_temp, y_temp
            st.session_state.splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
            st.success(f"✅ Split done. Train: {len(X_train)}, Val: {0 if X_val is None else len(X_val)}, Test: {len(X_test)}")

# ----------------------
# Tab 2: Diagnostics
# ----------------------
with tab2:
    st.subheader("Data Quality Diagnostics")
    df = st.session_state.df

    miss = df.isna().mean().sort_values(ascending=False)
    st.write("**Missing Value % per Column:**")
    st.dataframe(miss.to_frame("missing_pct"))
    st.bar_chart(miss)

    st.write(f"**Duplicate Rows:** {int(df.duplicated().sum())}")
    nunique = df.nunique()
    const_features = nunique[nunique <= 1]
    st.write("**Constant/Near-constant Features:**", const_features.to_dict())

    st.markdown("---")
    st.markdown("### 🔗 Multicollinearity (VIF)")
    num_cols = st.session_state.schema["numeric"]
    vif = _vif_frame(df, num_cols)
    if vif is not None:
        st.dataframe(vif)
        high_vif = vif[vif["VIF"] > 10]
        if not high_vif.empty:
            st.warning(f"⚠️ High VIF (>10): {high_vif.to_dict('records')}")
            st.info("💡 Tip: Consider PCA or dropping redundant features.")
    else:
        st.info("Not enough numeric columns for VIF analysis.")

# ----------------------
# Tab 3: Visualization
# ----------------------
with tab3:
    st.subheader("Interactive 2D/3D Visualization")
    df_all = st.session_state.df.copy()
    df_view = _sample_df(df_all)

    plot_type = st.selectbox("Plot Type", ["Scatter (2D)","Scatter (3D)","Histogram","Box","Violin","Line"], key="plot_type_select")
    cols = st.multiselect("Columns to Plot", list(df_view.columns), key="plot_cols_multi")
    color_opt = st.selectbox("Color By", ["None"] + list(df_view.columns), key="plot_color_select")
    size_opt = st.selectbox("Size By", ["None"] + list(df_view.columns), key="plot_size_select")
    hover_cols = st.multiselect("Tooltip Columns", list(df_view.columns), key="plot_hover_multi")
    link_enabled = st.checkbox("Enable Linked Filtering", value=True, key="linked_filter_check")

    if st.button("🎨 Render Plot", key="render_plot_btn"):
        color = None if color_opt == "None" else color_opt
        size_col = None if size_opt == "None" else size_opt

        if plot_type == "Scatter (2D)":
            if len(cols) < 2: st.warning("⚠️ Pick at least two columns.")
            else:
                df_plot = df_view.copy()
                if size_col and pd.api.types.is_numeric_dtype(df_plot[size_col]):
                    df_plot[size_col] = df_plot[size_col].fillna(df_plot[size_col].mean()).clip(lower=0)
                fig = px.scatter(df_plot, x=cols[0], y=cols[1], color=color, size=size_col, hover_data=hover_cols)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Scatter (3D)":
            if len(cols) < 3: st.warning("⚠️ Pick at least three columns.")
            else:
                df_plot = df_view.copy()
                if size_col:
                    if pd.api.types.is_numeric_dtype(df_plot[size_col]):
                        df_plot[size_col] = df_plot[size_col].fillna(df_plot[size_col].mean()).clip(lower=0)
                    else:
                        st.info("ℹ️ Size column is not numeric; ignoring size.")
                        size_col = None
                fig = px.scatter_3d(df_plot, x=cols[0], y=cols[1], z=cols[2], color=color, size=size_col)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Histogram":
            if len(cols) < 1: st.warning("⚠️ Pick one column.")
            else:
                fig = px.histogram(df_view, x=cols[0], color=color)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box":
            if len(cols) < 1: st.warning("⚠️ Pick one column.")
            else:
                fig = px.box(df_view, y=cols[0], color=color)
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Violin":
            if len(cols) < 1: st.warning("⚠️ Pick one column.")
            else:
                fig = px.violin(df_view, y=cols[0], color=color, box=True, points="all")
                st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Line":
            if len(cols) < 2: st.warning("⚠️ Pick x and y.")
            else:
                fig = px.line(df_view, x=cols[0], y=cols[1], color=color)
                st.plotly_chart(fig, use_container_width=True)

        if link_enabled:
            st.info("🔗 Linked filtering: apply numeric range filters below to subset data across tabs.")
            df_link = st.session_state.df.copy()
            filter_cols = [c for c in cols[:3] if c in st.session_state.schema["numeric"]]
            for c in filter_cols:
                rng = st.slider(f"Filter {c}", float(df_link[c].min()), float(df_link[c].max()),
                                (float(df_link[c].min()), float(df_link[c].max())), key=f"filter_slider_{c}")
                df_link = df_link[(df_link[c] >= rng[0]) & (df_link[c] <= rng[1])]
            st.session_state.selection_mask = df_link.index
            st.write("**Filtered Preview:**")
            st.dataframe(df_link.head(50))

# ----------------------
# Tab 4: Preprocessing
# ----------------------
with tab4:
    st.subheader("Preprocessing & Feature Engineering")

    if st.session_state.mode == "Modeling-safe" and (st.session_state.splits["train"] is None):
        st.warning("⚠️ Modeling-safe mode requires a train/val/test split before applying transforms. Use the Schema tab to split.")
    else:
        # Missing values
        st.markdown("### 🔧 Missing Value Handling")
        imp_cols = st.multiselect("Columns to Impute", list(st.session_state.df.columns), key="impute_cols_multi")
        imp_strategy = st.selectbox("Imputation Strategy", ["mean","median","most_frequent","constant","knn","ffill","bfill"], key="impute_strategy_select")
        fill_value = st.text_input("Constant Fill Value (if 'constant')", key="impute_fill_value")
        if st.button("Apply Imputation", key="apply_imputation_btn"):
            checkpoint()
            df_before = st.session_state.df.copy()
            df = st.session_state.df
            if st.session_state.mode == "Modeling-safe" and st.session_state.splits["train"] is not None:
                X_train, _y_train = st.session_state.splits["train"]
                if imp_strategy in ["ffill","bfill"]:
                    for c in imp_cols:
                        df[c] = df[c].ffill() if imp_strategy == "ffill" else df[c].bfill()
                elif imp_strategy == "knn":
                    knn = KNNImputer()
                    knn.fit(X_train[imp_cols])
                    df[imp_cols] = knn.transform(df[imp_cols])
                    st.session_state.fitted_transformers["imputer_knn"] = knn
                else:
                    imputer = SimpleImputer(strategy=("constant" if imp_strategy=="constant" else imp_strategy),
                                            fill_value=(fill_value if imp_strategy=="constant" else None))
                    imputer.fit(X_train[imp_cols])
                    df[imp_cols] = imputer.transform(df[imp_cols])
                    st.session_state.fitted_transformers["imputer"] = imputer
            else:
                if imp_strategy in ["ffill","bfill"]:
                    for c in imp_cols:
                        df[c] = df[c].ffill() if imp_strategy == "ffill" else df[c].bfill()
                elif imp_strategy == "knn":
                    knn = KNNImputer()
                    df[imp_cols] = knn.fit_transform(df[imp_cols])
                else:
                    imputer = SimpleImputer(strategy=("constant" if imp_strategy=="constant" else imp_strategy),
                                            fill_value=(fill_value if imp_strategy=="constant" else None))
                    df[imp_cols] = imputer.fit_transform(df[imp_cols])
            st.session_state.df = df
            st.session_state.schema = _infer_schema(df)
            log_op("impute", imp_cols, {"strategy": imp_strategy, "fill_value": fill_value})
            before_stats, after_stats = {}, {}
            for c in imp_cols:
                b, a = _stats_summary(df_before[c], df[c])
                before_stats[c], after_stats[c] = b, a
            st.session_state.last_explain = {"op":"impute","cols":imp_cols,"params":{"strategy":imp_strategy},"before":before_stats,"after":after_stats}
            st.success("✅ Imputation applied.")

        st.markdown("---")
        st.markdown("### 🏷️ Encoding")
        enc_cols = st.multiselect("Categorical Columns", st.session_state.schema["categorical"], key="encode_cols_multi")
        enc_method = st.selectbox("Encoding Method", ["onehot","ordinal","label","target_encoding (unsafe)"], key="encode_method_select")
        drop_first = st.checkbox("Drop First for One-Hot", value=False, key="encode_drop_first")
        target_col = st.selectbox("Target (for target encoding)", ["None"] + list(st.session_state.df.columns), key="encode_target_select")
        st.markdown("#### 🔻 Rare Category Handling")
        rare_threshold = st.slider("Rare Category Threshold (%)", 0.1, 5.0, 1.0, key="rare_threshold_slider")
        if st.button("Group Rare Categories", key="group_rare_btn"):
            checkpoint()
            df_before = st.session_state.df.copy()
            df = st.session_state.df
            changed = []
            for c in enc_cols:
                freq = df[c].value_counts(normalize=True)
                rares = freq[freq < (rare_threshold/100)].index
                if len(rares) > 0:
                    df[c] = df[c].apply(lambda x: "Other" if x in rares else x)
                    changed.append(c)
            st.session_state.df = df
            st.session_state.schema = _infer_schema(df)
            log_op("rare_group", changed, {"threshold_pct": rare_threshold})
            st.session_state.last_explain = {"op":"rare_group","cols":changed,"params":{"threshold_pct":rare_threshold},"before":{}, "after":{}}
            st.success(f"✅ Grouped rare categories in: {changed}")

        if st.session_state.mode == "Modeling-safe" and enc_method.startswith("target"):
            st.warning("⚠️ Target encoding is disabled in modeling-safe mode.")
        elif st.button("Apply Encoding", key="apply_encoding_btn"):
            checkpoint()
            df_before = st.session_state.df.copy()
            df = st.session_state.df
            if enc_cols:
                if enc_method == "onehot":
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=("first" if drop_first else None))
                    ohe.fit(df[enc_cols])
                    enc = ohe.transform(df[enc_cols])
                    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(enc_cols), index=df.index)
                    df = df.drop(columns=enc_cols).join(enc_df)
                    st.session_state.fitted_transformers["ohe"] = ohe
                elif enc_method == "ordinal":
                    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    df[enc_cols] = ord_enc.fit_transform(df[enc_cols])
                    st.session_state.fitted_transformers["ordinal"] = ord_enc
                elif enc_method == "label":
                    for c in enc_cols:
                        le = LabelEncoder()
                        df[c] = le.fit_transform(df[c].astype(str))
                        st.session_state.fitted_transformers[f"label_{c}"] = le
                elif enc_method.startswith("target") and target_col != "None":
                    st.warning("⚠️ Target encoding can cause leakage. Use with CV in modeling.")
                    for c in enc_cols:
                        means = df.groupby(c)[target_col].mean()
                        df[c] = df[c].map(means)
                st.session_state.df = df
                st.session_state.schema = _infer_schema(df)
                log_op("encode", enc_cols, {"method": enc_method, "drop_first": drop_first, "target": target_col})
                before_stats, after_stats = {}, {}
                for c in enc_cols:
                    if c in df_before.columns and c in st.session_state.df.columns:
                        b, a = _stats_summary(df_before[c], st.session_state.df[c])
                        before_stats[c], after_stats[c] = b, a
                st.session_state.last_explain = {"op":"encode","cols":enc_cols,"params":{"method":enc_method},"before":before_stats,"after":after_stats}
                st.success("✅ Encoding applied.")

        st.markdown("---")
        st.markdown("### 📏 Scaling / Normalization")
        scale_cols = st.multiselect("Columns to Scale (numeric)", st.session_state.schema["numeric"], key="scale_cols_multi")
        scale_method = st.selectbox("Scaler Type", ["standard","minmax","robust"], key="scale_method_select")
        if st.button("Apply Scaling", key="apply_scaling_btn"):
            checkpoint()
            df_before = st.session_state.df.copy()
            scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[scale_method]
            if st.session_state.mode == "Modeling-safe" and st.session_state.splits["train"] is not None:
                X_train, _ = st.session_state.splits["train"]
                scaler.fit(X_train[scale_cols])
                st.session_state.df[scale_cols] = scaler.transform(st.session_state.df[scale_cols])
            else:
                st.session_state.df[scale_cols] = scaler.fit_transform(st.session_state.df[scale_cols])
            st.session_state.fitted_transformers["scaler"] = scaler
            st.session_state.schema = _infer_schema(st.session_state.df)
            log_op("scale", scale_cols, {"method": scale_method})
            before_stats, after_stats = {}, {}
            for c in scale_cols:
                b, a = _stats_summary(df_before[c], st.session_state.df[c])
                before_stats[c], after_stats[c] = b, a
            st.session_state.last_explain = {"op":"scale","cols":scale_cols,"params":{"method":scale_method},"before":before_stats,"after":after_stats}
            st.success("✅ Scaling applied.")

        st.markdown("---")
        st.markdown("### 🔄 Value Transformations")
        trans_cols = st.multiselect("Columns to Transform (numeric)", st.session_state.schema["numeric"], key="transform_cols_multi")
        trans_method = st.selectbox("Transformation Type", ["log","boxcox","yeo-johnson"], key="transform_method_select")
        if st.button("Apply Transformation", key="apply_transform_btn"):
            checkpoint()
            df_before = st.session_state.df.copy()
            df = st.session_state.df
            if trans_method == "log":
                for c in trans_cols:
                    df[c] = np.log1p(df[c].clip(lower=0))
            elif trans_method == "boxcox":
                for c in trans_cols:
                    x = df[c].clip(lower=1e-6)
                    df[c], _ = boxcox(x)
            elif trans_method == "yeo-johnson":
                pt = PowerTransformer(method="yeo-johnson")
                df[trans_cols] = pt.fit_transform(df[trans_cols])
                st.session_state.fitted_transformers["power"] = pt
            st.session_state.df = df
            st.session_state.schema = _infer_schema(df)
            log_op("transform", trans_cols, {"method": trans_method})
            before_stats, after_stats = {}, {}
            for c in trans_cols:
                b, a = _stats_summary(df_before[c], st.session_state.df[c])
                before_stats[c], after_stats[c] = b, a
            st.session_state.last_explain = {"op":"transform","cols":trans_cols,"params":{"method":trans_method},"before":before_stats,"after":after_stats}
            st.success("✅ Transformation applied.")

        st.markdown("---")
        st.markdown("### 💡 Recommendations & Warnings")
        tips = []
        for c in st.session_state.schema["categorical"]:
            card = st.session_state.df[c].nunique(dropna=True)
            if card > 20:
                tips.append(f"Column '{c}' has high cardinality ({card}). Prefer one-hot with handle_unknown or careful target encoding with CV.")
        heavy_skew = [c for c in st.session_state.schema["numeric"] if abs(skew(pd.to_numeric(st.session_state.df[c], errors='coerce').dropna())) > 1]
        if heavy_skew:
            tips.append(f"Heavy skew detected in {heavy_skew}. Consider log/Box-Cox/Yeo-Johnson.")
        if st.session_state.target:
            vc = st.session_state.df[st.session_state.target].value_counts(normalize=True, dropna=True)
            if len(vc) > 1 and vc.min() < 0.1:
                tips.append("Target imbalance detected. Prefer stratified splits and sampling strategies.")
        tips.append("If correlation |r| > 0.9 or VIF > 10, consider PCA or removing redundant features.")
        for t in tips:
            st.info(t)

# ----------------------
# Tab 5: Dimensionality reduction
# ----------------------
with tab5:
    st.subheader("Structure Discovery")
    if st.session_state.mode == "Modeling-safe" and st.session_state.splits["train"] is None:
        st.warning("⚠️ PCA disabled until train/val/test split in modeling-safe mode.")
    else:
        st.markdown("### 🎯 PCA")
        pca_cols = st.multiselect("Columns for PCA", st.session_state.schema["numeric"], key="pca_cols")
        n_comp = st.number_input("Number of Components", min_value=2, max_value=10, value=2, step=1, key="pca_n_comp")
        if st.button("Run PCA", key="run_pca_btn"):
            checkpoint()
            if len(pca_cols) < 2:
                st.warning("Select at least two numeric columns for PCA.")
                st.stop()
            X = st.session_state.df[pca_cols]
            X_clean = X.dropna()
            if X_clean.shape[0] < n_comp:
                st.warning("Not enough complete rows for the selected number of components.")
                st.stop()
            pca = PCA(n_components=n_comp, random_state=42)
            pcs = pca.fit_transform(X_clean)
            for i in range(n_comp):
                st.session_state.df.loc[X_clean.index, f"PC{i+1}"] = pcs[:, i]
            st.session_state.schema = _infer_schema(st.session_state.df)
            log_op(
                "reduce",
                pca_cols,
                {
                    "method": "pca",
                    "n_components": n_comp,
                    "explained_variance": pca.explained_variance_ratio_.tolist()
                }
            )
            st.write(
                "**Explained Variance:**",
                [f"PC{i+1}: {v:.2%}" for i, v in enumerate(pca.explained_variance_ratio_)]
            )
            st.success("✅ PCA components added.")
    st.markdown("---")
    st.markdown("### 🗺️ t-SNE / UMAP")
    dr_method = st.selectbox("Dimensionality Reduction Method", ["t-SNE","UMAP"], key="dr_method_select")
    dr_cols = st.multiselect("Select Columns", st.session_state.schema["numeric"], key="dr_cols")
    perplexity = st.slider("t-SNE Perplexity", 5, 50, 30, key="tsne_perplexity")
    n_neighbors = st.slider("UMAP Neighbors", 5, 100, 15, key="umap_neighbors")
    min_dist = st.slider("UMAP Min Distance", 0.0, 1.0, 0.1, key="umap_min_dist")
    if st.button("Run Dimensionality Reduction", key="run_dr_btn"):
        checkpoint()
        X = st.session_state.df[dr_cols].dropna()
        X_view = _sample_df(X, max_rows=3000)
        if dr_method == "t-SNE":
            emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(X_view)
            st.session_state.df.loc[X_view.index, "TSNE1"] = emb[:,0]
            st.session_state.df.loc[X_view.index, "TSNE2"] = emb[:,1]
            log_op("reduce", dr_cols, {"method": "tsne", "perplexity": perplexity})
        else:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            emb = reducer.fit_transform(X_view)
            st.session_state.df.loc[X_view.index, "UMAP1"] = emb[:,0]
            st.session_state.df.loc[X_view.index, "UMAP2"] = emb[:,1]
            log_op("reduce", dr_cols, {"method": "umap", "n_neighbors": n_neighbors, "min_dist": min_dist})
        st.session_state.schema = _infer_schema(st.session_state.df)
        st.success(f"✅ {dr_method} embedding added.")

# ----------------------
# Tab 6: Correlation & stats
# ----------------------
with tab6:
    st.subheader("Correlation & Statistical Analysis")
    df = st.session_state.df
    method = st.selectbox("Correlation Method", ["pearson","spearman","kendall"], key="corr_method_select")
    num_cols = st.session_state.schema["numeric"]
    if not num_cols:
        st.info("ℹ️ No numeric columns for correlation.")
    else:
        corr = df[num_cols].corr(method=method)
        st.dataframe(corr)
        fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        high_pairs = []
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                val = corr.loc[c1, c2]
                if abs(val) > 0.9:
                    high_pairs.append((c1, c2, round(val,3)))
        if high_pairs:
            st.warning(f"⚠️ High correlation pairs (>|0.9|): {high_pairs}")

    st.markdown("---")
    st.markdown("### 📊 Groupby Aggregations")
    group_cols = st.multiselect("Group By Columns", list(df.columns), key="groupby_cols_multi")
    agg_col = st.selectbox("Aggregate Column (numeric)", num_cols, key="groupby_agg_col") if num_cols else None
    agg_fn = st.selectbox("Aggregation Function", ["mean","median","sum","count","std"], key="groupby_agg_fn")
    if st.button("Run Groupby", key="run_groupby_btn"):
        if group_cols and agg_col:
            res = getattr(df.groupby(group_cols)[agg_col], agg_fn)()
            st.dataframe(res)

    st.markdown("---")
    st.markdown("### 🧪 Hypothesis Testing (Two-Sample t-test and KS)")
    test_col = st.selectbox("Numeric Column for Test", num_cols, key="hyp_test_col") if num_cols else None
    cat_col = st.selectbox("Category to Split By", st.session_state.schema["categorical"], key="hyp_cat_col") if st.session_state.schema["categorical"] else None
    if st.button("Run Tests", key="run_hyp_tests_btn"):
        if not test_col or not cat_col:
            st.warning("⚠️ Select both a numeric column and a categorical split.")
        else:
            groups = df[cat_col].dropna().unique()
            if len(groups) < 2:
                st.warning("⚠️ Need at least two groups.")
            else:
                g1 = df[df[cat_col] == groups[0]][test_col].dropna()
                g2 = df[df[cat_col] == groups[1]][test_col].dropna()
                tstat, tp = ttest_ind(g1, g2, equal_var=False)
                ks, kp = ks_2samp(g1, g2)
                st.write({"t_stat": float(tstat), "t_pvalue": float(tp), "ks_stat": float(ks), "ks_pvalue": float(kp)})

# ----------------------
# Tab 7: History & versions
# ----------------------
with tab7:
    st.subheader("Pipeline History & Versions")
    st.json(st.session_state.pipeline)

    ver_name = st.text_input("Save Current Dataset as Version (name)", key="version_name_input")
    if st.button("Save Version", key="save_version_btn"):
        if ver_name.strip():
            save_version(ver_name.strip())
            st.success(f"✅ Saved version '{ver_name.strip()}'")

    ver_select = st.selectbox("Load Version", ["None"] + list(st.session_state.versions.keys()), key="version_select")
    if ver_select != "None":
        if st.button("Load Selected Version", key="load_version_btn"):
            load_version(ver_select)
            st.success(f"✅ Loaded version '{ver_select}'")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Undo Last Operation", key="undo_btn"):
            undo_last()
    with c2:
        if st.button("➡️ Redo Last Undone", key="redo_btn"):
            redo_last()

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Replay")
    uploaded_pipeline = st.file_uploader("Upload Pipeline JSON/YAML", type=["json","yaml","yml"], key="pipeline_uploader")
    strict = st.checkbox("Strict Mode (fail on mismatches)", value=False, key="strict_mode_check")
    if uploaded_pipeline and st.button("Apply Uploaded Pipeline", key="apply_pipeline_btn"):
        try:
            if uploaded_pipeline.name.lower().endswith(".json"):
                pipeline = json.load(uploaded_pipeline)
            else:
                pipeline = yaml.safe_load(uploaded_pipeline)
            checkpoint()
            new_df = apply_pipeline(pipeline, df=st.session_state.df.copy(), strict=strict)
            st.session_state.df = new_df
            st.session_state.schema = _infer_schema(st.session_state.df)
            st.success("✅ Pipeline applied.")
        except Exception as e:
            st.error(f"❌ Failed to apply pipeline: {e}")

# ----------------------
# Tab 8: Export
# ----------------------
with tab8:
    st.subheader("Export Options")

    st.markdown("#### 📥 Dataset Export")
    csv_bytes = st.session_state.df.to_csv(index=False).encode()
    st.markdown(_download_link_bytes(csv_bytes, "modified.csv", "text/csv", "📥 Download Modified CSV"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📦 Pipeline Export")
    pipeline_json = json.dumps(st.session_state.pipeline, indent=2).encode()
    st.markdown(_download_link_bytes(pipeline_json, "pipeline.json", "application/json", "📥 Download Pipeline (JSON)"), unsafe_allow_html=True)
    pipeline_yaml = yaml.dump(st.session_state.pipeline).encode()
    st.markdown(_download_link_bytes(pipeline_yaml, "pipeline.yaml", "text/yaml", "📥 Download Pipeline (YAML)"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Auto-EDA Report")
    if st.button("Generate Simple EDA Report (HTML)", key="gen_eda_report_btn"):
        summary = st.session_state.df.describe(include="all").to_html()
        miss = st.session_state.df.isna().mean().to_frame("missing_pct").to_html()
        html = f"""
        <html><head><meta charset="utf-8"><title>EDA Report</title></head>
        <body>
        <h1>EDA Report</h1>
        <h2>Summary Statistics</h2>{summary}
        <h2>Missingness</h2>{miss}
        <h2>Schema</h2><pre>{st.session_state.schema}</pre>
        <h2>Pipeline</h2><pre>{json.dumps(st.session_state.pipeline, indent=2)}</pre>
        </body></html>
        """
        st.download_button("📥 Download report.html", data=html.encode(), file_name="report.html", mime="text/html", key="download_report_btn")

    if PPROF_AVAILABLE and st.button("Generate Rich EDA Profile (pandas-profiling)", key="gen_profile_btn"):
        profile = pandas_profiling.ProfileReport(st.session_state.df, title="EDA Profile", explorative=True)
        html = profile.to_html()
        st.download_button("📥 Download profile.html", data=html.encode(), file_name="profile.html", mime="text/html", key="download_profile_btn")

# ----------------------
# Tab 9: Validation & explainability
# ----------------------
with tab9:
    st.subheader("Data Validation Rules")
    df = st.session_state.df

    rule_type = st.selectbox("Rule Type", ["Range", "Regex", "Datetime monotonicity", "Category whitelist"], key="rule_type_select")
    validation_col = st.selectbox("Select Column", list(df.columns), key="validation_col_select")
    violations = []
    
    if rule_type == "Range":
        min_val = st.text_input("Minimum Value (optional)", key="range_min_input")
        max_val = st.text_input("Maximum Value (optional)", key="range_max_input")
        if st.button("Check Range", key="check_range_btn"):
            series = pd.to_numeric(df[validation_col], errors="coerce")
            if min_val.strip():
                try:
                    mn = float(min_val)
                    violations.extend(df.index[series < mn].tolist())
                except: st.error("❌ Invalid minimum value.")
            if max_val.strip():
                try:
                    mx = float(max_val)
                    violations.extend(df.index[series > mx].tolist())
                except: st.error("❌ Invalid maximum value.")
            st.write({"violations_count": len(set(violations))})
            if violations:
                st.dataframe(df.loc[sorted(set(violations))][[validation_col]].head(50))
                
    elif rule_type == "Regex":
        pattern = st.text_input("Regex Pattern (e.g., ^[A-Za-z]+$)", key="regex_pattern_input")
        if st.button("Check Regex", key="check_regex_btn"):
            try:
                regex = re.compile(pattern)
                series = df[validation_col].astype(str)
                violations = df.index[~series.str.match(regex)].tolist()
                st.write({"violations_count": len(violations)})
                if violations:
                    st.dataframe(df.loc[violations][[validation_col]].head(50))
            except re.error:
                st.error("❌ Invalid regex pattern.")
                
    elif rule_type == "Datetime monotonicity":
        if st.button("Check Monotonic Increasing", key="check_mono_btn"):
            if not pd.api.types.is_datetime64_any_dtype(df[validation_col]):
                st.error("❌ Column is not datetime type.")
            else:
                dt = pd.to_datetime(df[validation_col], errors="coerce")
                mono = dt.is_monotonic_increasing
                st.write({"monotonic_increasing": bool(mono)})
                if not mono:
                    st.info("💡 Consider sorting or fixing timestamps.")
                    
    elif rule_type == "Category whitelist":
        whitelist_raw = st.text_area("Allowed Categories (comma-separated)", key="whitelist_input")
        if st.button("Check Whitelist", key="check_whitelist_btn"):
            allowed = [x.strip() for x in whitelist_raw.split(",") if x.strip()]
            series = df[validation_col].astype(str)
            violations = df.index[~series.isin(allowed)].tolist()
            st.write({"violations_count": len(violations)})
            if violations:
                st.dataframe(df.loc[violations][[validation_col]].head(50))

    st.markdown("---")
    st.subheader("✨ Explainability for Last Transformation")
    le = st.session_state.last_explain
    if not le:
        st.info("ℹ️ Apply a preprocessing step to see before/after statistics.")
    else:
        st.write({"operation": le["op"], "columns": le["cols"], "params": le["params"]})
        for c in le["cols"]:
            b = le["before"].get(c)
            a = le["after"].get(c)
            if b and a:
                st.write(f"**{c}:** mean {b['mean']:.4f} → {a['mean']:.4f}, std {b['std']:.4f} → {a['std']:.4f}, skew {b['skew']:.4f} → {a['skew']:.4f}, kurtosis {b['kurtosis']:.4f} → {a['kurtosis']:.4f}")
                if c in st.session_state.df.columns:
                    fig = px.histogram(st.session_state.df, x=c, nbins=30, title=f"Distribution after {le['op']}: {c}")
                    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
