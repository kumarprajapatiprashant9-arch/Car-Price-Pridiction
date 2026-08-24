import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 45px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 30px;
}

.price-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #ddd;
    margin-top: 20px;
}

.price {
    font-size: 40px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")


# =========================================================
# CHECK FILES
# =========================================================

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ car_price_model.pkl file nahi mili!")
    st.info("Pehle terminal me ye command run karo:")
    st.code("python train_model.py")
    st.stop()


try:
    data = load_data()
except FileNotFoundError:
    st.error("❌ car_data.csv file nahi mili!")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚗 Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Used Car Price Prediction System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🚘 Enter Car Details")

st.sidebar.write(
    "Enter the information about your car."
)


# =========================================================
# GET UNIQUE VALUES
# =========================================================

car_names = sorted(data["Car_Name"].unique())

companies = sorted(data["Company"].unique())

fuel_types = sorted(data["Fuel_Type"].unique())

selling_types = sorted(data["Selling_Type"].unique())

transmissions = sorted(data["Transmission"].unique())


# =========================================================
# CAR NAME
# =========================================================

car_name = st.sidebar.selectbox(
    "🚘 Car Name",
    car_names
)


# =========================================================
# COMPANY
# =========================================================

company = st.sidebar.selectbox(
    "🏢 Company",
    companies
)


# =========================================================
# YEAR
# =========================================================

year = st.sidebar.number_input(
    "📅 Manufacturing Year",
    min_value=2000,
    max_value=2026,
    value=2020,
    step=1
)


# =========================================================
# PRESENT PRICE
# =========================================================

present_price = st.sidebar.number_input(
    "💰 Current/Original Price (₹ Lakh)",
    min_value=0.5,
    max_value=100.0,
    value=10.0,
    step=0.5
)


# =========================================================
# KILOMETERS
# =========================================================

kms_driven = st.sidebar.number_input(
    "🛣️ Kilometers Driven",
    min_value=0,
    max_value=1000000,
    value=25000,
    step=1000
)


# =========================================================
# FUEL TYPE
# =========================================================

fuel_type = st.sidebar.selectbox(
    "⛽ Fuel Type",
    fuel_types
)


# =========================================================
# SELLING TYPE
# =========================================================

selling_type = st.sidebar.selectbox(
    "🏷️ Selling Type",
    selling_types
)


# =========================================================
# TRANSMISSION
# =========================================================

transmission = st.sidebar.selectbox(
    "⚙️ Transmission",
    transmissions
)


# =========================================================
# OWNER
# =========================================================

owner = st.sidebar.selectbox(
    "👤 Number of Previous Owners",
    [0, 1, 2, 3]
)


# =========================================================
# MAIN SECTION
# =========================================================

st.subheader("📋 Selected Car Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🚘 Car",
        car_name
    )

    st.metric(
        "🏢 Company",
        company
    )


with col2:

    st.metric(
        "📅 Year",
        year
    )

    st.metric(
        "⛽ Fuel",
        fuel_type
    )


with col3:

    st.metric(
        "⚙️ Transmission",
        transmission
    )

    st.metric(
        "🛣️ Kilometers",
        f"{kms_driven:,}"
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Car Price",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create input dataframe

    input_data = pd.DataFrame({
        "Car_Name": [car_name],
        "Company": [company],
        "Year": [year],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_type],
        "Selling_Type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner]
    })


    # Make prediction

    try:

        prediction = model.predict(input_data)[0]

        # Avoid negative price

        prediction = max(0, prediction)


        # Convert lakh to rupees

        price_in_rupees = prediction * 100000


        # Success message

        st.success(
            "✅ Car price prediction completed!"
        )


        # Display result

        st.markdown(
            f"""
            <div class="price-box">

                <h2>💰 Estimated Selling Price</h2>

                <div class="price">
                    ₹ {prediction:.2f} Lakh
                </div>

                <p>
                    Approximately ₹ {price_in_rupees:,.0f}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Additional information

        st.markdown("---")

        st.subheader("📊 Prediction Details")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Original Price",
                f"₹ {present_price:.2f} Lakh"
            )


        with col2:

            st.metric(
                "Predicted Price",
                f"₹ {prediction:.2f} Lakh"
            )


        with col3:

            difference = present_price - prediction

            st.metric(
                "Price Difference",
                f"₹ {difference:.2f} Lakh"
            )


    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"
        )


# =========================================================
# DATASET
# =========================================================

st.markdown("---")

st.subheader("📊 Dataset")


with st.expander("View Complete Dataset"):

    st.dataframe(
        data,
        use_container_width=True
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write("""
    ### 🚗 Car Price Prediction

    This project predicts the estimated selling price of a
    used car using Machine Learning.

    ### Features Used

    - Car Name
    - Company
    - Manufacturing Year
    - Present Price
    - Kilometers Driven
    - Fuel Type
    - Selling Type
    - Transmission
    - Previous Owners

    ### Machine Learning Algorithm

    Random Forest Regression

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Joblib
    - Streamlit
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🚗 Car Price Prediction | "
    "Python + Machine Learning + Streamlit"
)