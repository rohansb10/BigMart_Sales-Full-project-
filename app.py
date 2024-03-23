import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
file_path = r"C:\\Users\\Rohan\\Pictures\\rohan\\BigMart_Sales project\\gradient_boosting_model.pkl"
loaded_model = pickle.load(open(file_path, 'rb'))
scaler =  pickle.load(open(r"C:\\Users\\Rohan\\Pictures\\rohan\\BigMart_Sales project\\scaler.pkl", 'rb'))
# Mappings for categorical variables
fat_content_mapping = {'Low Fat': 1, 'Regular': 2, 'LF': 1, 'reg': 2, 'low fat': 1}
item_type_mapping = {'Fruits and Vegetables': 1, 'Snack Foods': 2, 'Household': 3, 'Frozen Foods': 4, 'Dairy': 5,
                     'Canned': 6, 'Baking Goods': 7, 'Health and Hygiene': 8, 'Soft Drinks': 9, 'Meat': 10,
                     'Breads': 11, 'Hard Drinks': 12, 'Others': 13, 'Starchy Foods': 14, 'Breakfast': 15, 'Seafood': 16}
outlet_size_mapping = {'Medium': 1, 'Undefined': 2, 'Small': 3, 'High': 4}
outlet_type_mapping = {'Supermarket': 1, 'Grocery Store': 2}

# Get user input
user_input = {
    'Item_Fat_Content': input("Enter Item Fat Content (e.g., 'Low Fat' or 'Regular'): "),
    'Item_Type': input("Enter Item Type: "),
    'Outlet_Size': input("Enter Outlet Size: "),
    'Outlet_Type': input("Enter Outlet Type: "),
    'Item_MRP': float(input("Enter Item MRP: "))
}

user_input_df = pd.DataFrame([user_input])
def preprocess_input(input_data, mappings, scaler):
    # Apply mappings to respective columns
    input_data['Item_Fat_Content'] = mappings['Item_Fat_Content'].get(input_data['Item_Fat_Content'], 0)
    input_data['Item_Type'] = mappings['Item_Type'].get(input_data['Item_Type'], 0)
    input_data['Outlet_Size'] = mappings['Outlet_Size'].get(input_data['Outlet_Size'], 0)
    input_data['Outlet_Type'] = mappings['Outlet_Type'].get(input_data['Outlet_Type'], 0)

    return input_data
# Preprocess user input using mappings
preprocessed_input = preprocess_input(user_input_df.iloc[0].to_dict(), {
    'Item_Fat_Content': fat_content_mapping,
    'Item_Type': item_type_mapping,
    'Outlet_Size': outlet_size_mapping,
    'Outlet_Type': outlet_type_mapping
}, scaler)

# Convert the preprocessed input to a DataFrame
preprocessed_input_df = pd.DataFrame([preprocessed_input])

# Use the fitted scaler to transform the data
preprocessed_input_scaled = scaler.transform(preprocessed_input_df)

# Make predictions
prediction = loaded_model.predict(preprocessed_input_scaled)

print("Predicted Sales:", prediction[0])
