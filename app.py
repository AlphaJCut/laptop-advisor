import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Smart Laptop Advisor",
    page_icon="laptop",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================== SESSION STATE FOR CURRENCY =====================
if "inr_to_vnd_rate" not in st.session_state:
    st.session_state.inr_to_vnd_rate = 287


# ===================== CURRENCY FUNCTIONS =====================
def inr_to_vnd(price_inr):
    """Convert price from INR to VND."""
    return price_inr * st.session_state.inr_to_vnd_rate


def format_vnd(price_vnd):
    """Format VND price with dot separator."""
    return f"{price_vnd:,.0f}".replace(",", ".")


def format_price(price_inr, currency="VND"):
    """Format price based on currency type."""
    if currency == "VND":
        price_vnd = inr_to_vnd(price_inr)
        return f"{format_vnd(price_vnd)} VND"
    else:
        return f"{price_inr:,.0f} INR"


# ===================== CUSTOM CSS =====================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .laptop-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===================== LOAD MODELS =====================
@st.cache_resource
def load_all_models():
    """Load all models with caching."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    data_dir = os.path.join(os.path.dirname(__file__), "data", "processed")

    models = {}

    # Load preprocessor
    preprocessor_path = os.path.join(data_dir, "preprocessor.pkl")
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, "rb") as f:
            models["preprocessor"] = pickle.load(f)

    # Load price model
    price_model_path = os.path.join(model_dir, "price_model.pkl")
    if os.path.exists(price_model_path):
        with open(price_model_path, "rb") as f:
            models["price_model"] = pickle.load(f)

    # Load processed data
    data_path = os.path.join(data_dir, "processed_data.csv")
    if os.path.exists(data_path):
        models["data"] = pd.read_csv(data_path)

    return models


# ===================== CONSTANTS =====================
BRANDS = [
    "HP",
    "Acer",
    "Lenovo",
    "Dell",
    "Asus",
    "Apple",
    "MSI",
    "Samsung",
    "Gigabyte",
    "Infinix",
    "Realme",
]

PROCESSORS = [
    "Dual Core, 4 Threads",
    "Quad Core, 8 Threads",
    "Hexa Core, 12 Threads",
    "Octa Core, 8 Threads",
    "Octa Core (4P + 4E), 12 Threads",
    "10 Cores (2P + 8E), 12 Threads",
    "12 Cores (4P + 8E), 16 Threads",
    "14 Cores (6P + 8E), 20 Threads",
]

GPUS = [
    "Intel UHD Graphics",
    "Intel Iris Xe Graphics",
    "AMD Radeon Graphics",
    "NVIDIA GeForce MX450",
    "4GB NVIDIA GeForce RTX 2050",
    "4GB NVIDIA GeForce RTX 3050",
    "6GB NVIDIA GeForce RTX 3060",
    "6GB NVIDIA GeForce RTX 4050",
    "8GB NVIDIA GeForce RTX 4060",
    "8GB NVIDIA GeForce RTX 4070",
]

RAM_OPTIONS = [4, 8, 16, 32, 64]
STORAGE_OPTIONS = [128, 256, 512, 1024, 2048]
SCREEN_SIZES = [13.3, 14.0, 15.6, 16.0, 17.3]
OS_OPTIONS = ["Windows 11 OS", "Windows 10 OS", "Mac OS", "Linux", "Chrome OS"]

# Use case profiles for recommendations
USE_CASE_PROFILES = {
    "gaming": {
        "name": "Gaming",
        "description": "AAA games, high graphics",
        "weights": {
            "gpu_score": 0.35,
            "processor_score": 0.25,
            "ram_gb": 0.2,
            "performance_score": 0.2,
        },
        "min_requirements": {"ram_gb": 16, "gpu_score": 5},
    },
    "office": {
        "name": "Office Work",
        "description": "Word, Excel, email, web browsing",
        "weights": {
            "processor_score": 0.3,
            "ram_gb": 0.3,
            "storage_gb": 0.2,
            "screen_size": 0.2,
        },
        "min_requirements": {"ram_gb": 8},
    },
    "creative": {
        "name": "Creative Work",
        "description": "Photoshop, Premiere, design",
        "weights": {
            "ram_gb": 0.3,
            "processor_score": 0.25,
            "gpu_score": 0.25,
            "storage_gb": 0.2,
        },
        "min_requirements": {"ram_gb": 16, "processor_score": 6},
    },
    "student": {
        "name": "Student",
        "description": "Study, research, budget-friendly",
        "weights": {
            "ram_gb": 0.25,
            "storage_gb": 0.25,
            "processor_score": 0.25,
            "screen_size": 0.25,
        },
        "min_requirements": {"ram_gb": 8},
    },
    "developer": {
        "name": "Programming",
        "description": "Coding, Docker, VMs",
        "weights": {
            "ram_gb": 0.35,
            "processor_score": 0.3,
            "storage_gb": 0.2,
            "screen_size": 0.15,
        },
        "min_requirements": {"ram_gb": 16, "processor_score": 6},
    },
    "portable": {
        "name": "Portable",
        "description": "Lightweight, long battery",
        "weights": {
            "screen_size": -0.3,
            "processor_score": 0.3,
            "ram_gb": 0.2,
            "storage_gb": 0.2,
        },
        "min_requirements": {"ram_gb": 8},
    },
}


# ===================== HELPER FUNCTIONS =====================
def create_price_gauge(predicted_price, currency, min_price=None, max_price=None):
    """Create a gauge chart for price visualization."""
    if currency == "VND":
        value = inr_to_vnd(predicted_price)
        prefix = ""
        suffix = " VND"
        if min_price is None:
            min_price = 3000000
        if max_price is None:
            max_price = max(60000000, value * 1.2)
        steps = [
            {"range": [3000000, 10000000], "color": "#E8F5E9"},
            {"range": [10000000, 20000000], "color": "#C8E6C9"},
            {"range": [20000000, 35000000], "color": "#FFF9C4"},
            {"range": [35000000, 60000000], "color": "#FFCDD2"},
        ]
    else:
        value = predicted_price
        prefix = ""
        suffix = " INR"
        if min_price is None:
            min_price = 10000
        if max_price is None:
            max_price = max(200000, value * 1.2)
        steps = [
            {"range": [10000, 30000], "color": "#E8F5E9"},
            {"range": [30000, 60000], "color": "#C8E6C9"},
            {"range": [60000, 100000], "color": "#FFF9C4"},
            {"range": [100000, 200000], "color": "#FFCDD2"},
        ]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font": {"size": 32},
                "valueformat": ",.0f",
            },
            title={"text": "Predicted Price", "font": {"size": 18}},
            gauge={
                "axis": {"range": [min_price, max_price], "tickformat": ",.0f"},
                "bar": {"color": "#1E88E5"},
                "steps": steps,
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def calculate_match_score(laptop_row, use_case):
    """Calculate how well a laptop matches a use case."""
    if use_case not in USE_CASE_PROFILES:
        return 0

    profile = USE_CASE_PROFILES[use_case]

    # Check minimum requirements
    min_reqs = profile.get("min_requirements", {})
    for feature, min_val in min_reqs.items():
        if feature in laptop_row and laptop_row[feature] < min_val:
            return 0  # Disqualified

    # Calculate weighted score
    weights = profile["weights"]
    score = 0
    total_weight = 0

    for feature, weight in weights.items():
        if feature in laptop_row:
            value = laptop_row[feature]
            # Normalize value
            if feature == "ram_gb":
                normalized = min(value / 32, 1)
            elif feature == "storage_gb":
                normalized = min(value / 1024, 1)
            elif feature == "screen_size":
                normalized = (value - 13) / (17 - 13)
            elif feature in ["processor_score", "gpu_score", "performance_score"]:
                normalized = value / 10
            else:
                normalized = 0.5

            score += weight * normalized
            total_weight += abs(weight)

    if total_weight > 0:
        return (score / total_weight) * 100
    return 0


def get_recommendations(df, use_case, budget, top_n=5, currency="VND"):
    """Get laptop recommendations based on use case and budget."""
    df_filtered = df.copy()

    # Find price column
    price_col = "price" if "price" in df_filtered.columns else "Price"

    # Convert budget to INR for filtering
    if currency == "VND":
        budget_inr = budget / st.session_state.inr_to_vnd_rate
    else:
        budget_inr = budget

    # Filter by budget
    if price_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[price_col] <= budget_inr]

    if len(df_filtered) == 0:
        return pd.DataFrame()

    # Calculate match scores
    df_filtered["match_score"] = df_filtered.apply(
        lambda row: calculate_match_score(row, use_case), axis=1
    )

    # Filter out disqualified laptops
    df_filtered = df_filtered[df_filtered["match_score"] > 0]

    if len(df_filtered) == 0:
        return pd.DataFrame()

    # Sort by match score
    recommendations = df_filtered.nlargest(top_n, "match_score")

    return recommendations


def prepare_user_input_simple(user_specs, preprocessor):
    """Prepare user input for prediction."""
    df_input = pd.DataFrame([user_specs])

    df_input["ram_gb"] = user_specs["ram_gb"]
    df_input["storage_gb"] = user_specs["storage_gb"]
    df_input["screen_size"] = user_specs["screen_size"]
    df_input["weight_kg"] = 2.0

    # Processor score
    processor = user_specs["processor"].lower()
    if "14 core" in processor or "16 core" in processor:
        processor_score = 9
    elif "12 core" in processor:
        processor_score = 8
    elif "10 core" in processor or "octa" in processor:
        processor_score = 7
    elif "hexa" in processor:
        processor_score = 6
    elif "quad" in processor:
        processor_score = 5
    else:
        processor_score = 4
    df_input["processor_score"] = processor_score

    # GPU score
    gpu = user_specs["gpu"].lower()
    if "4070" in gpu or "4080" in gpu:
        gpu_score = 9
    elif "4060" in gpu or "4050" in gpu:
        gpu_score = 7
    elif "3060" in gpu or "3070" in gpu:
        gpu_score = 6
    elif "3050" in gpu or "2050" in gpu:
        gpu_score = 5
    elif "mx" in gpu:
        gpu_score = 4
    elif "iris" in gpu:
        gpu_score = 3
    else:
        gpu_score = 2
    df_input["gpu_score"] = gpu_score

    # Performance score
    df_input["performance_score"] = (
        processor_score * 0.4
        + gpu_score * 0.3
        + (user_specs["ram_gb"] / 64) * 10 * 0.2
        + (user_specs["storage_gb"] / 2048) * 10 * 0.1
    )

    df_input["portability_score"] = 6.0
    df_input["total_pixels"] = 1920 * 1080
    df_input["ppi"] = np.sqrt(1920 * 1080) / user_specs["screen_size"]
    df_input["is_gaming"] = 1 if "rtx" in gpu or "gtx" in gpu else 0
    df_input["is_ultraportable"] = 1 if user_specs["screen_size"] <= 14 else 0
    df_input["hdd"] = 0
    df_input["ssd"] = user_specs["storage_gb"]

    # Encode categorical features
    label_encoders = preprocessor.get("label_encoders", {})
    categorical_cols = [
        "brand",
        "product",
        "processor",
        "gpu",
        "os",
        "Cpu_brand",
        "Gpu_brand",
    ]

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            if col == "brand":
                val = user_specs["brand"]
            elif col == "processor":
                val = user_specs["processor"]
            elif col == "gpu":
                val = user_specs["gpu"]
            elif col == "os":
                val = user_specs["os"]
            else:
                val = "Unknown"

            if val in le.classes_:
                df_input[f"{col}_encoded"] = le.transform([val])[0]
            else:
                df_input[f"{col}_encoded"] = 0
        else:
            df_input[f"{col}_encoded"] = 0

    feature_columns = preprocessor.get("feature_columns", [])

    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    X = df_input[feature_columns].values

    scaler = preprocessor.get("scaler")
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    return X_scaled


# ===================== MAIN APP =====================
def main():
    # Header
    st.markdown(
        '<h1 class="main-header">Smart Laptop Advisor</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">AI-powered Price Prediction and Recommendations</p>',
        unsafe_allow_html=True,
    )

    # ===================== SIDEBAR =====================
    st.sidebar.header("Settings")

    # Currency selection
    currency = st.sidebar.radio(
        "Currency:",
        options=["VND", "INR"],
        index=0,
        help="VND = Vietnamese Dong, INR = Indian Rupee",
    )

    # Exchange rate info
    st.sidebar.info(f"Exchange Rate: 1 INR = {st.session_state.inr_to_vnd_rate:,} VND")

    # Custom exchange rate
    custom_rate = st.sidebar.checkbox("Custom exchange rate")
    if custom_rate:
        new_rate = st.sidebar.number_input(
            "1 INR = ? VND:",
            min_value=100,
            max_value=500,
            value=st.session_state.inr_to_vnd_rate,
            step=10,
        )
        st.session_state.inr_to_vnd_rate = new_rate

    st.sidebar.divider()
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown("""
    - Dataset: Kaggle (India)
    - Algorithm: Random Forest
    - Accuracy: R2 = 0.83
    """)

    # Load models
    try:
        models = load_all_models()
        if not models:
            st.error("Could not load models. Please train models first.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

    # Create 3 tabs
    tab1, tab2, tab3 = st.tabs(
        ["Price Prediction", "Smart Recommendations", "Market Analysis"]
    )

    # =================== TAB 1: PRICE PREDICTION ===================
    with tab1:
        st.header("Predict Laptop Price")
        st.write("Configure laptop specifications to get an estimated price.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Specs")
            brand = st.selectbox("Brand", BRANDS, index=0, key="pred_brand")
            processor = st.selectbox(
                "Processor (CPU)", PROCESSORS, index=4, key="pred_cpu"
            )
            gpu = st.selectbox("Graphics Card (GPU)", GPUS, index=0, key="pred_gpu")
            os_choice = st.selectbox(
                "Operating System", OS_OPTIONS, index=0, key="pred_os"
            )

        with col2:
            st.subheader("Performance and Display")
            ram = st.select_slider("RAM (GB)", RAM_OPTIONS, value=8, key="pred_ram")
            storage = st.select_slider(
                "Storage SSD (GB)", STORAGE_OPTIONS, value=512, key="pred_storage"
            )
            screen_size = st.select_slider(
                "Screen Size (inches)", SCREEN_SIZES, value=15.6, key="pred_screen"
            )

        # Predict button
        if st.button("Predict Price", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                user_specs = {
                    "brand": brand,
                    "processor": processor,
                    "ram_gb": ram,
                    "storage_gb": storage,
                    "gpu": gpu,
                    "screen_size": screen_size,
                    "os": os_choice,
                }

                try:
                    X = prepare_user_input_simple(user_specs, models["preprocessor"])
                    price_model = models["price_model"]["model"]
                    predicted_price_inr = price_model.predict(X)[0]
                    predicted_price_vnd = inr_to_vnd(predicted_price_inr)

                    st.success("Prediction complete!")

                    col_result1, col_result2 = st.columns([1, 2])

                    with col_result1:
                        if currency == "VND":
                            st.metric(
                                label="Predicted Price",
                                value=f"{format_vnd(predicted_price_vnd)} VND",
                            )
                            st.caption(f"Approximately {predicted_price_inr:,.0f} INR")

                            margin_vnd = predicted_price_vnd * 0.15
                            st.info(
                                f"**Price Range:** {format_vnd(predicted_price_vnd - margin_vnd)} - {format_vnd(predicted_price_vnd + margin_vnd)} VND"
                            )

                            if predicted_price_vnd < 10000000:
                                category = "Budget"
                            elif predicted_price_vnd < 18000000:
                                category = "Mid-Range"
                            elif predicted_price_vnd < 30000000:
                                category = "High-End"
                            else:
                                category = "Premium"
                        else:
                            st.metric(
                                label="Predicted Price",
                                value=f"{predicted_price_inr:,.0f} INR",
                            )
                            st.caption(
                                f"Approximately {format_vnd(predicted_price_vnd)} VND"
                            )

                            margin_inr = predicted_price_inr * 0.15
                            st.info(
                                f"**Price Range:** {predicted_price_inr - margin_inr:,.0f} - {predicted_price_inr + margin_inr:,.0f} INR"
                            )

                            if predicted_price_inr < 30000:
                                category = "Budget"
                            elif predicted_price_inr < 50000:
                                category = "Mid-Range"
                            elif predicted_price_inr < 80000:
                                category = "High-End"
                            else:
                                category = "Premium"

                        st.write(f"**Category:** {category}")

                    with col_result2:
                        fig = create_price_gauge(predicted_price_inr, currency)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

    # =================== TAB 2: RECOMMENDATIONS ===================
    with tab2:
        st.header("Smart Laptop Recommendations")
        st.write("Select your use case and budget to get personalized recommendations.")

        if "data" not in models:
            st.warning(
                "No data available for recommendations. Please train model first."
            )
        else:
            df = models["data"]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Use Case")

                use_case = st.radio(
                    "Select your primary use:",
                    options=list(USE_CASE_PROFILES.keys()),
                    format_func=lambda x: (
                        f"{USE_CASE_PROFILES[x]['name']} - {USE_CASE_PROFILES[x]['description']}"
                    ),
                    index=0,
                )

            with col2:
                st.subheader("Budget & Results")

                if currency == "VND":
                    budget = st.slider(
                        "Maximum Budget (VND):",
                        min_value=5000000,
                        max_value=100000000,
                        value=25000000,
                        step=1000000,
                        format="%d",
                    )
                    st.caption(
                        f"Approximately {budget / st.session_state.inr_to_vnd_rate:,.0f} INR"
                    )
                else:
                    budget = st.slider(
                        "Maximum Budget (INR):",
                        min_value=15000,
                        max_value=300000,
                        value=80000,
                        step=5000,
                        format="%d",
                    )
                    st.caption(
                        f"Approximately {format_vnd(budget * st.session_state.inr_to_vnd_rate)} VND"
                    )

                top_n = st.slider(
                    "Number of results to show:",
                    min_value=3,
                    max_value=10,
                    value=5,
                )

            # Get recommendations button
            if st.button("Find Laptops", type="primary", use_container_width=True):
                with st.spinner("Searching..."):
                    recommendations = get_recommendations(
                        df, use_case, budget, top_n, currency
                    )

                    if len(recommendations) == 0:
                        st.warning(f"""
                        No laptops found matching your criteria:
                        - Use Case: {USE_CASE_PROFILES[use_case]["name"]}
                        - Budget: {format_price(budget / st.session_state.inr_to_vnd_rate if currency == "VND" else budget, currency)}
                        
                        Suggestion: Try increasing your budget or selecting a different use case.
                        """)
                    else:
                        st.success(
                            f"Found {len(recommendations)} matching laptops (showing top {top_n})!"
                        )

                        # Find column names
                        price_col = (
                            "price" if "price" in recommendations.columns else "Price"
                        )
                        brand_col = (
                            "brand" if "brand" in recommendations.columns else "Company"
                        )
                        product_col = (
                            "product"
                            if "product" in recommendations.columns
                            else "Product"
                        )

                        # Display recommendations
                        for idx, (_, laptop) in enumerate(
                            recommendations.iterrows(), 1
                        ):
                            with st.container():
                                col_info, col_score = st.columns([4, 1])

                                with col_info:
                                    brand_name = laptop.get(
                                        brand_col, laptop.get("brand", "Unknown")
                                    )
                                    product_name = laptop.get(
                                        product_col, laptop.get("Product", "Unknown")
                                    )
                                    price_inr = laptop.get(
                                        price_col, laptop.get("Price", 0)
                                    )

                                    st.markdown(
                                        f"### {idx}. {brand_name} - {product_name[:50]}..."
                                    )

                                    spec_col1, spec_col2, spec_col3 = st.columns(3)

                                    with spec_col1:
                                        ram = laptop.get(
                                            "ram_gb", laptop.get("Ram", "N/A")
                                        )
                                        storage = laptop.get(
                                            "storage_gb", laptop.get("SSD", "N/A")
                                        )
                                        st.write(f"**RAM:** {ram} GB")
                                        st.write(f"**SSD:** {storage} GB")

                                    with spec_col2:
                                        processor = laptop.get(
                                            "processor", laptop.get("Cpu", "N/A")
                                        )
                                        if (
                                            isinstance(processor, str)
                                            and len(processor) > 30
                                        ):
                                            processor = processor[:30] + "..."
                                        screen = laptop.get(
                                            "screen_size", laptop.get("Inches", "N/A")
                                        )
                                        st.write(f"**CPU:** {processor}")
                                        st.write(f"**Screen:** {screen} inches")

                                    with spec_col3:
                                        gpu = laptop.get(
                                            "gpu", laptop.get("Gpu", "N/A")
                                        )
                                        if isinstance(gpu, str) and len(gpu) > 25:
                                            gpu = gpu[:25] + "..."
                                        st.write(f"**GPU:** {gpu}")

                                        price_display = format_price(
                                            price_inr, currency
                                        )
                                        st.write(f"**Price:** {price_display}")

                                with col_score:
                                    match_score = laptop.get("match_score", 0)
                                    st.metric("Match", f"{match_score:.0f}%")

                                st.divider()

    # =================== TAB 3: MARKET ANALYSIS ===================
    with tab3:
        st.header("Market Analysis")
        st.write("Explore laptop market trends and insights from the dataset.")

        if "data" in models:
            df = models["data"]

            price_col = "price" if "price" in df.columns else "Price"
            brand_col = "brand" if "brand" in df.columns else "Company"

            # Add VND column
            if price_col in df.columns:
                df["price_vnd"] = df[price_col] * st.session_state.inr_to_vnd_rate

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Laptops", f"{len(df):,}")
            with col2:
                if currency == "VND":
                    avg_price = df[price_col].mean() * st.session_state.inr_to_vnd_rate
                    st.metric("Avg Price", f"{format_vnd(avg_price)} VND")
                else:
                    st.metric("Avg Price", f"{df[price_col].mean():,.0f} INR")
            with col3:
                if brand_col in df.columns:
                    st.metric("Brands", df[brand_col].nunique())
            with col4:
                if currency == "VND":
                    min_p = df[price_col].min() * st.session_state.inr_to_vnd_rate
                    max_p = df[price_col].max() * st.session_state.inr_to_vnd_rate
                    st.metric(
                        "Price Range", f"{min_p / 1e6:.0f}M - {max_p / 1e6:.0f}M VND"
                    )
                else:
                    st.metric(
                        "Price Range",
                        f"{df[price_col].min():,.0f} - {df[price_col].max():,.0f} INR",
                    )

            st.divider()

            # Charts
            if price_col in df.columns and brand_col in df.columns:
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    if currency == "VND":
                        fig1 = px.box(
                            df,
                            x=brand_col,
                            y="price_vnd",
                            color=brand_col,
                            title="Price Distribution by Brand (VND)",
                        )
                        fig1.update_yaxes(title_text="Price (VND)")
                    else:
                        fig1 = px.box(
                            df,
                            x=brand_col,
                            y=price_col,
                            color=brand_col,
                            title="Price Distribution by Brand (INR)",
                        )
                        fig1.update_yaxes(title_text="Price (INR)")
                    fig1.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig1, use_container_width=True)

                with col_chart2:
                    if currency == "VND":
                        brand_avg = (
                            df.groupby(brand_col)["price_vnd"]
                            .mean()
                            .sort_values(ascending=False)
                            .head(10)
                        )
                        title = "Average Price by Brand - Top 10 (VND)"
                    else:
                        brand_avg = (
                            df.groupby(brand_col)[price_col]
                            .mean()
                            .sort_values(ascending=False)
                            .head(10)
                        )
                        title = "Average Price by Brand - Top 10 (INR)"

                    fig2 = px.bar(x=brand_avg.index, y=brand_avg.values, title=title)
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)

            # RAM vs Price
            if "ram_gb" in df.columns and price_col in df.columns:
                col_chart3, col_chart4 = st.columns(2)

                with col_chart3:
                    y_col = "price_vnd" if currency == "VND" else price_col
                    y_label = "Price (VND)" if currency == "VND" else "Price (INR)"

                    fig3 = px.scatter(
                        df,
                        x="ram_gb",
                        y=y_col,
                        color=brand_col,
                        title="Price vs RAM",
                        labels={"ram_gb": "RAM (GB)", y_col: y_label},
                    )
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)

                with col_chart4:
                    if "performance_score" in df.columns:
                        fig4 = px.scatter(
                            df,
                            x="performance_score",
                            y=y_col,
                            title="Price vs Performance Score",
                            labels={
                                "performance_score": "Performance Score",
                                y_col: y_label,
                            },
                        )
                        fig4.update_layout(height=400)
                        st.plotly_chart(fig4, use_container_width=True)

            # Correlation heatmap
            st.subheader("Feature Correlations")
            numeric_cols = [
                "ram_gb",
                "storage_gb",
                "screen_size",
                "processor_score",
                "gpu_score",
                "performance_score",
                price_col,
            ]
            existing_cols = [col for col in numeric_cols if col in df.columns]

            if len(existing_cols) > 1:
                corr_matrix = df[existing_cols].corr()
                fig5 = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Matrix",
                )
                fig5.update_layout(height=500)
                st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("No data available for analysis.")

    # Footer
    st.divider()


if __name__ == "__main__":
    main()
