import os
import zipfile
import time
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.methods import group_by_color, group_by_phash, group_by_ssim, group_by_orb, group_by_cnn
from utils.benchmark import benchmark_methods

# --- Streamlit UI ---
st.title("Image Similarity Detection System")

# Upload Section
upload_type = st.radio("Choose upload type:", ("Individual Images", "Zip Archive"))
image_paths = []
temp_dir = "temp_images"
os.makedirs(temp_dir, exist_ok=True)

if upload_type == "Individual Images":
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(file_path)
else:
    uploaded_zip = st.file_uploader("Upload zip archive", type="zip")
    if uploaded_zip:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.png'))]

# Ground Truth Upload
ground_truth_file = st.file_uploader("Upload ground truth CSV (optional)", type="csv")
ground_truth_df = pd.read_csv(ground_truth_file) if ground_truth_file else None

# Parameters
st.sidebar.header("Threshold Settings")
hist_threshold = st.sidebar.slider("Color Histogram Threshold", 0.0, 1.0, 0.9)
phash_threshold = st.sidebar.slider("pHash Max Distance", 0, 64, 10)
ssim_threshold = st.sidebar.slider("SSIM Threshold", 0.0, 1.0, 0.9)
orb_match_threshold = st.sidebar.slider("ORB Match Threshold", 0, 200, 50)
cnn_threshold = st.sidebar.slider("CNN Similarity Threshold", 0.0, 1.0, 0.9)

method = st.selectbox("Select Method", ["Color Histogram", "pHash", "SSIM", "ORB", "CNN"])
run_benchmark = st.checkbox("Run Benchmark for All Methods")

# Method Map
method_map = {
    "Color Histogram": group_by_color,
    "pHash": group_by_phash,
    "SSIM": group_by_ssim,
    "ORB": group_by_orb,
    "CNN": group_by_cnn,
}

# Method Parameters
method_kwargs = {
    "Color Histogram": {"threshold": hist_threshold},
    "pHash": {"max_distance": phash_threshold},
    "SSIM": {"ssim_threshold": ssim_threshold},
    "ORB": {"match_threshold": orb_match_threshold},
    "CNN": {"threshold": cnn_threshold},
}

# --- Process Images ---
if st.button("Process Images") and image_paths:
    st.write("Processing images...")
    progress_bar = st.progress(0)

    if not run_benchmark:
        func = method_map[method]
        kwargs = method_kwargs[method]
        with st.spinner(f"Running {method}..."):
            start_time = time.time()
            groups = func(image_paths, **kwargs)
            execution_time = time.time() - start_time

        st.write(f"**Execution Time**: {execution_time:.2f} seconds")
        st.write(f"**Number of Groups Found**: {len(groups)}")
        for i, group in enumerate(groups):
            st.subheader(f"Group {i+1}")
            cols = st.columns(min(len(group), 5))
            for j, path in enumerate(group[:5]):
                cols[j].image(path, use_container_width=True)
        progress_bar.progress(100)

    else:
        with st.spinner("Running benchmarks..."):
            selected_methods = {k: v for k, v in method_map.items()}
            results = benchmark_methods(selected_methods, image_paths, **method_kwargs)

        # Compile results
        rows = []
        for name, data in results.items():
            groups = data["groups"]
            time_taken = data["time"]
            precision = "N/A"

            if ground_truth_df is not None:
                ground_truth_pairs = set(
                    tuple(sorted((row["image_1"], row["image_2"])))
                    for _, row in ground_truth_df.iterrows()
                    if row["is_duplicate"]
                )
                predicted_pairs = set()
                for group in groups:
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            predicted_pairs.add(tuple(sorted((group[i], group[j]))))
                tp = len(predicted_pairs & ground_truth_pairs)
                precision = 100 * tp / len(predicted_pairs) if predicted_pairs else 0

            rows.append({
                "Method": name,
                "Time (sec)": round(time_taken, 2),
                "Precision (%)": precision,
                "Num Groups": len(groups)
            })

        results_df = pd.DataFrame(rows)
        st.write("**Benchmark Results**")
        st.dataframe(results_df)

        # Plots
        fig, ax = plt.subplots()
        sns.barplot(x="Method", y="Time (sec)", data=results_df, ax=ax)
        ax.set_title("Execution Time Comparison")
        st.pyplot(fig)
        plt.close()

        if ground_truth_df is not None:
            fig, ax = plt.subplots()
            sns.barplot(x="Method", y="Precision (%)", data=results_df, ax=ax)
            ax.set_title("Precision Comparison")
            st.pyplot(fig)
            plt.close()

        progress_bar.progress(100)

# Clean-up
if st.button("Clear Temporary Files"):
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    st.write("Temporary files cleared.")
