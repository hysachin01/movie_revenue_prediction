
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Movie Success Predictor", page_icon="🎬")
st.title("🎬 Movie Success Prediction App")
st.markdown("Predict the revenue of a movie based on its budget, runtime, and number of votes.")

# ✅ Load the trained model correctly
try:
    model = joblib.load('best_movie_model.pkl')
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 🔢 Input features (only 3)
budget = st.number_input("💰 Budget (in millions)", min_value=0.0, step=1.0)
runtime = st.slider("⏱️ Runtime (minutes)", 60, 180, 120)
votes = st.number_input("🗳️ Number of Votes", min_value=0, step=100)

# 🚀 Predict button
if st.button("Predict Revenue"):
    input_data = np.array([[budget, runtime, votes]])  # only 3 features
    try:
        prediction = model.predict(input_data)
        st.success(f"🎉 Predicted Revenue: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
