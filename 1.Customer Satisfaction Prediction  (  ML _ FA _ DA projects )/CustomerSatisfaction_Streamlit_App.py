# open jupyter terminal
# chnage directory where "CustomerSatisfactionApp.py" is kept
# streamlit run CustomerSatisfactionApp.py


import pickle
import numpy as np
import streamlit as st

model = pickle.load(open(r'E:\PYTHONCLASSTF\UnifiedMentorProjects\DA & DS\1.Customer Satisfaction Prediction  (  ML _ FA _ DA projects )\customer_satisfaction.pkl', 'rb'))

label_encoders = pickle.load(open(r'E:\PYTHONCLASSTF\UnifiedMentorProjects\DA & DS\1.Customer Satisfaction Prediction  (  ML _ FA _ DA projects )\label_encoders.pkl', 'rb'))

st.title('Customer Satisfaction App')
st.write('This app predicts the customer rating based on various features using a random forest model')

cust_age = st.number_input('Customer Age:', min_value=18.0, max_value=80.0, value=18.0, step=1.0)
cust_gender = st.selectbox('Customer Gender:', ('Male', 'Female', 'Other'))

prod_purchased = st.selectbox('Product Purchased:', ('Canon EOS', 'GoPro Hero', 'Nest Thermostat', 'Philips Hue Lights', 'Amazon Echo', 'LG Smart TV', 'Sony Xperia', 'Roomba Robot Vacuum', 'Apple AirPods', 'LG OLED', 'iPhone', 'Sony 4K HDR TV', 'Garmin Forerunner', 'LG Washing Machine', 'Canon DSLR Camera', 'Nikon D', 'Nintendo Switch Pro Controller', 'Google Pixel', 'Fitbit Charge', 'Sony PlayStation', 'HP Pavilion', 'Microsoft Office', 'Amazon Kindle', 'Google Nest', 'Dyson Vacuum Cleaner', 'Bose SoundLink Speaker', 'Autodesk AutoCAD', 'Microsoft Xbox Controller', 'Samsung Galaxy', 'PlayStation', 'Fitbit Versa Smartwatch', 'Microsoft Surface', 'Bose QuietComfort', 'Samsung Soundbar', 'Xbox', 'Asus ROG', 'MacBook Pro', 'Dell XPS', 'Lenovo ThinkPad', 'GoPro Action Camera', 'Adobe Photoshop', 'Nintendo Switch'))

ticket_type = st.selectbox('Ticket Type:', ('Refund request', 'Technical issue', 'Cancellation request', 'Product inquiry', 'Billing inquiry'))

ticket_sub = st.selectbox('Ticket Subject:', ('Refund request', 'Software bug', 'Product compatibility', 'Delivery problem', 'Hardware issue', 'Battery life', 'Network problem', 'Installation support', 'Product setup', 'Payment issue', 'Product recommendation', 'Account access', 'Peripheral compatibility', 'Data loss', 'Cancellation request', 'Display issue'))

ticket_status = st.selectbox('Ticket Status:', ('Closed'))

ticket_priority = st.selectbox('Ticket Priority:', ('Medium', 'Critical', 'High', 'Low'))

ticket_channel = st.selectbox('Ticket Channel:', ('Email', 'Phone', 'Social media', 'Chat'))

if st.button('Predict Customer Ratings'):
    encoded_gender = label_encoders['Customer Gender'].transform([cust_gender])[0]
    encoded_product = label_encoders['Product Purchased'].transform([prod_purchased])[0]
    encoded_ticket_type = label_encoders['Ticket Type'].transform([ticket_type])[0]
    encoded_ticket_sub = label_encoders['Ticket Subject'].transform([ticket_sub])[0]
    encoded_ticket_status = label_encoders['Ticket Status'].transform([ticket_status])[0]
    encoded_ticket_priority = label_encoders['Ticket Priority'].transform([ticket_priority])[0]
    encoded_ticket_channel = label_encoders['Ticket Channel'].transform([ticket_channel])[0]

    input_array = np.array([[cust_age, encoded_gender, encoded_product, encoded_ticket_type,
                             encoded_ticket_sub, encoded_ticket_status, encoded_ticket_priority,
                             encoded_ticket_channel]])
    prediction = model.predict(input_array)
    st.success(f'The predicted customer rating is: {prediction[0]}')                    