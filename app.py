import os
import zipfile
import time
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.methods import group_by_color, group_by_ahash, group_by_ssim, group_by_orb, group_by_cnn, compare_by_color, compare_by_ahash, compare_by_ssim, compare_by_orb, compare_by_cnn
from utils.benchmark import benchmark_methods

# --- Streamlit UI ---
st.title("Image Similarity Detection System")

# Upload Section
upload_type = st.radio("Choose upload type:", ("Compare Two Images", "Individual Images", "Zip Archive"))
image_paths = []
temp_dir = "temp_images"
os.makedirs(temp_dir, exist_ok=True)

if upload_type == "Compare Two Images":
    col1, col2 = st.columns(2)
    with col1:
        img1 = st.file_uploader("Upload First Image", type=["jpg", "png"], key="img1")
    with col2:
        img2 = st.file_uploader("Upload Second Image", type=["jpg", "png"], key="img2")
    if img1 and img2:
        img1_path = os.path.join(temp_dir, img1.name)
        img2_path = os.path.join(temp_dir, img2.name)
        with open(img1_path, "wb") as f:
            f.write(img1.getbuffer())
        with open(img2_path, "wb") as f:
            f.write(img2.getbuffer())
        image_paths = [img1_path, img2_path]
elif upload_type == "Individual Images":
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

# Parameters
st.sidebar.header("Threshold Settings")
hist_threshold = st.sidebar.slider("Color Histogram Threshold", 0.0, 1.0, 0.9)
ahash_threshold = st.sidebar.slider("aHash Max Distance", 0, 64, 10)
ssim_threshold = st.sidebar.slider("SSIM Threshold", 0.0, 1.0, 0.9)
orb_match_threshold = st.sidebar.slider("ORB Match Threshold", 0, 200, 50)
cnn_threshold = st.sidebar.slider("CNN Similarity Threshold", 0.0, 1.0, 0.9)

method = st.selectbox("Select Method", ["Color Histogram", "aHash", "SSIM", "ORB", "CNN"])
run_benchmark = st.checkbox("Run Benchmark for All Methods", disabled=upload_type == "Compare Two Images")

# Method Map for Grouping
method_map = {
    "Color Histogram": group_by_color,
    "aHash": group_by_ahash,
    "SSIM": group_by_ssim,
    "ORB": group_by_orb,
    "CNN": group_by_cnn
}

# Method Map for Comparing Two Images
compare_method_map = {
    "Color Histogram": compare_by_color,
    "aHash": compare_by_ahash,
    "SSIM": compare_by_ssim,
    "ORB": compare_by_orb,
    "CNN": compare_by_cnn
}

# Method Parameters
method_kwargs = {
    "Color Histogram": {"threshold": hist_threshold},
    "aHash": {"max_distance": ahash_threshold},
    "SSIM": {"ssim_threshold": ssim_threshold},
    "ORB": {"match_threshold": orb_match_threshold},
    "CNN": {"threshold": cnn_threshold}
}

# --- Process Images ---
if st.button("Process Images") and image_paths:
    st.write("Processing images...")
    progress_bar = st.progress(0)

    if upload_type == "Compare Two Images":
        func = compare_method_map[method]
        kwargs = method_kwargs[method]
        with st.spinner(f"Running {method}..."):
            start_time = time.time()
            similarity = func(image_paths[0], image_paths[1], **kwargs)
            execution_time = time.time() - start_time

        st.write(f"**Execution Time**: {execution_time:.3f} seconds")
        st.write(f"**Similarity Score**: {similarity:.3f}")
        if method == "aHash":
            st.write("Images are similar" if similarity <= ahash_threshold else "Images are different")
        else:
            st.write("Images are similar" if similarity >= method_kwargs[method].get("threshold", 0.9) else "Images are different")
        
        cols = st.columns(2)
        cols[0].image(image_paths[0], caption="Image 1", use_container_width=True)
        cols[1].image(image_paths[1], caption="Image 2", use_container_width=True)
        progress_bar.progress(100)

    elif not run_benchmark:
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
            rows.append({
                "Method": name,
                "Time (sec)": round(time_taken, 2),
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
        progress_bar.progress(100)

# Clean-up
if st.button("Clear Temporary Files"):
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    st.write("Temporary files cleared.")