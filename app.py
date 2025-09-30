uploaded_file = st.file_uploader("Upload Parkinson's dataset (.csv)", type="csv")
if uploaded_file:
    X, y = load_data(uploaded_file)
