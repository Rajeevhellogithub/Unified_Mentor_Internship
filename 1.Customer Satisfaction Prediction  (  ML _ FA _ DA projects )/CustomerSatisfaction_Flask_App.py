from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoders
model_path = r'E:\PYTHONCLASSTF\UnifiedMentorProjects\DA & DS\1.Customer Satisfaction Prediction  (  ML _ FA _ DA projects )\customer_satisfaction.pkl'
label_encoder_path = r'E:\PYTHONCLASSTF\UnifiedMentorProjects\DA & DS\1.Customer Satisfaction Prediction  (  ML _ FA _ DA projects )\label_encoders.pkl'

try:
    model = pickle.load(open(model_path, 'rb'))
    label_encoders = pickle.load(open(label_encoder_path, 'rb'))
except FileNotFoundError:
    print("Error: Model or label encoder file not found. Please check the paths.")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            cust_age = float(request.form['cust_age'])
            cust_gender = request.form['cust_gender']
            prod_purchased = request.form['prod_purchased']
            ticket_type = request.form['ticket_type']
            ticket_sub = request.form['ticket_sub']
            ticket_status = request.form['ticket_status']
            ticket_priority = request.form['ticket_priority']
            ticket_channel = request.form['ticket_channel']

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
            prediction = model.predict(input_array)[0]
            prediction = f'{prediction:.2f}'  # Format prediction
        except Exception as e:
            prediction = f"Error during prediction: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)