import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
file_path_model = r"C:\\Users\\Rohan\\Pictures\\rohan\\BigMart_Sales project\\gradient_boosting_model.pkl"
file_path_scaler = r"C:\\Users\\Rohan\\Pictures\\rohan\\BigMart_Sales project\\scaler.pkl"

loaded_model = pickle.load(open(file_path_model, 'rb'))
scaler = pickle.load(open(file_path_scaler, 'rb'))

# Mappings for categorical variables
fat_content_mapping = {'Low Fat': 1, 'Regular': 2, 'LF': 1, 'reg': 2, 'low fat': 1}
item_type_mapping = {'Fruits and Vegetables': 1, 'Snack Foods': 2, 'Household': 3, 'Frozen Foods': 4, 'Dairy': 5,
                     'Canned': 6, 'Baking Goods': 7, 'Health and Hygiene': 8, 'Soft Drinks': 9, 'Meat': 10,
                     'Breads': 11, 'Hard Drinks': 12, 'Others': 13, 'Starchy Foods': 14, 'Breakfast': 15, 'Seafood': 16}
outlet_size_mapping = {'Medium': 1, 'Undefined': 2, 'Small': 3, 'High': 4}
outlet_type_mapping = {'Supermarket': 1, 'Grocery Store': 2}

# Streamlit App
st.title('BigMart Sales Prediction App')

# Get user input through Streamlit widgets
item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
item_type = st.text_input("Item Type")
outlet_size = st.selectbox("Outlet Size", ['Medium', 'Undefined', 'Small', 'High'])
outlet_type = st.selectbox("Outlet Type", ['Supermarket', 'Grocery Store'])
item_mrp = st.number_input("Item MRP", value=0.0)

# Create a DataFrame with user input
user_input = {
    'Item_Fat_Content': item_fat_content,
    'Item_Type': item_type,
    'Outlet_Size': outlet_size,
    'Outlet_Type': outlet_type,
    'Item_MRP': item_mrp
}

user_input_df = pd.DataFrame([user_input])

def preprocess_input(input_data, mappings):
    # Apply mappings to respective columns
    input_data['Item_Fat_Content'] = mappings['Item_Fat_Content'].get(input_data['Item_Fat_Content'], 0)
    input_data['Item_Type'] = mappings['Item_Type'].get(input_data['Item_Type'], 0)
    input_data['Outlet_Size'] = mappings['Outlet_Size'].get(input_data['Outlet_Size'], 0)
    input_data['Outlet_Type'] = mappings['Outlet_Type'].get(input_data['Outlet_Type'], 0)

    return input_data

# Preprocess user input using mappings
try:
    preprocessed_input = preprocess_input(user_input_df.iloc[0].to_dict(), {
        'Item_Fat_Content': fat_content_mapping,
        'Item_Type': item_type_mapping,
        'Outlet_Size': outlet_size_mapping,
        'Outlet_Type': outlet_type_mapping
    })
except KeyError as e:
    st.error(f"Error: {e}. Please check your input.")
    st.stop()

# Convert the preprocessed input to a DataFrame
preprocessed_input_df = pd.DataFrame([preprocessed_input])

# Use the fitted scaler to transform the data
preprocessed_input_scaled = scaler.transform(preprocessed_input_df)

# Predict button
if st.button('Predict'):
    # Make predictions with compatibility handling
    try:
        prediction = loaded_model.predict(preprocessed_input_scaled)
        # Display prediction
        st.subheader("Predicted Sales:")
        st.write(prediction[0])
    except AttributeError as e:
        if 'get_init_raw_predictions' in str(e):
            st.warning("Compatibility issue with model. Consider retraining the model.")
        else:
            st.error(f"AttributeError: {e}. Handling compatibility issue...")
