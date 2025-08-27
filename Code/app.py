from flask import Flask, render_template, request, redirect, url_for, session
import os
import csv
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime  # Ensure datetime is imported for date handling
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = os.urandom(24)

users = {'user': 'password'}
user_codes = {}  # Dictionary to store user-generated codes

csv_file_path = r"C:\Users\moham\Downloads\dataset.csv"

if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Customers', 'Quantity', 'Date', 'Total Payment in TND'])

banks = [
    "Arab Tunisian Bank (ATB)",
    "Banque Nationale Agricole (BNA)",
    "Attijari Bank",
    "Banque de Tunisie (BT)",
    "Amen Bank (AB)",
    "Banque Internationale Arabe de Tunisie (BIAT)",
    "La Poste Tunisienne",
    "Banque de l'Habitat (BH)",
    "Arab Banking Corporation (ABC)",
    "Banque Tuniso-Libyenne (BTL)"
]

data = pd.read_csv(r"C:\Users\moham\Downloads\maintenance_events_1000.csv")

data['date'] = pd.to_datetime(data['date'])

data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

le_bank = LabelEncoder()
data['bank_code'] = le_bank.fit_transform(data['bank'])

le_town = LabelEncoder()
data['town_code'] = le_town.fit_transform(data['town'])

X_bank_town = data[['day_of_week', 'month', 'year']]
y_bank = data['bank_code']
y_town = data['town_code']

X_technicians = data[['day_of_week', 'month', 'year', 'bank_code', 'town_code']]
y_technicians = data['num_technicians']

X_train_bank_town, X_test_bank_town, y_bank_train, y_bank_test, y_town_train, y_town_test = train_test_split(
    X_bank_town, y_bank, y_town, test_size=0.2, random_state=42)

X_train_technicians, X_test_technicians, y_tech_train, y_tech_test = train_test_split(
    X_technicians, y_technicians, test_size=0.2, random_state=42)

model_bank = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_bank.fit(X_train_bank_town, y_bank_train)

model_town = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_town.fit(X_train_bank_town, y_town_train)

model_technicians = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_technicians.fit(X_train_technicians, y_tech_train)

def predict_failure(date_input):
    # Convert input date to datetime
    date_input = datetime.strptime(date_input, '%Y-%m-%d')
    
    # Extract features from input date
    day_of_week = date_input.weekday()
    month = date_input.month
    year = date_input.year
    
    # Predict bank and town failure
    bank_code_pred = model_bank.predict([[day_of_week, month, year]])[0]
    town_code_pred = model_town.predict([[day_of_week, month, year]])[0]
    
    predicted_bank = le_bank.inverse_transform([bank_code_pred])[0]
    predicted_town = le_town.inverse_transform([town_code_pred])[0]
    
    num_technicians_pred = model_technicians.predict([[day_of_week, month, year, bank_code_pred, town_code_pred]])[0]
    
    return predicted_bank, predicted_town, (math.ceil(num_technicians_pred)-2)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('loggedin'))
        else:
            return render_template('login.html', message='Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        code = request.form['code']
        new_username = request.form['username']
        new_password = request.form['password']
        
        # Check if the code is valid and matches the one generated for the current user
        if code in user_codes and user_codes[code] == session['username']:
            # Add new user to the simulated database (replace with actual logic)
            users[new_username] = new_password
            return redirect(url_for('login'))
        else:
            return render_template('register.html', message='Invalid code. Please try again.')

    return render_template('register.html')

@app.route('/loggedin')
def loggedin():
    if 'username' in session:
        return render_template('loggedin.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/getcode', methods=['GET', 'POST'])
def getcode():
    if 'username' in session:
        if request.method == 'POST':
            # Generate a unique code for the current user
            code = str(random.randint(1000, 9999)) + session['username']
            user_codes[code] = session['username']  # Store the generated code
            return render_template('getcode.html', code=code, show_continue_button=True)
        else:
            return render_template('getcode.html')
    else:
        return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    # Read data from CSV file and pass it to the template
    data = []
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    return render_template('dashboard.html', data=data)

@app.route('/predictdata', methods=['GET', 'POST'])
def predictdata():
    if request.method == 'POST':
        input_date = request.form['input_date']
        predicted_bank, predicted_town, num_technicians_pred = predict_failure(input_date)
        prediction = (predicted_bank, predicted_town, num_technicians_pred)
        return render_template('predictdata.html', prediction=prediction)
    return render_template('predictdata.html')

@app.route('/updatedata', methods=['GET', 'POST'])
def updatedata():
    if request.method == 'POST':
        bank = request.form['bank']
        quantity = request.form['quantity']
        date = request.form['date']

        # Append data to the CSV file
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([bank, quantity, date])

        # Data visualization code starts here
        # Load the updated dataset
        df = pd.read_csv(r"C:\Users\moham\Downloads\dataset.csv")

        # PIE CHART
        quantity_per_bank = df.groupby('Customers')['Quantity'].sum()
        plt.figure(figsize=(10, 8))
        pie = quantity_per_bank.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        colors = plt.cm.tab20c(range(len(quantity_per_bank)))
        for label, color in zip(pie.patches, colors):
            label.set_color(color)
        plt.title('Nombre des GABS achetés par Client')
        plt.ylabel('')

        # Save the pie chart
        pie_chart_path = r'static/output1.png'
        plt.savefig(pie_chart_path)
        print(f"Pie chart saved to '{pie_chart_path}'.")

        # BAR CHART
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        quantities_per_year = df.groupby('Year')['Quantity'].sum()
        plt.figure(figsize=(12, 6))
        plt.bar(quantities_per_year.index, quantities_per_year.values, align='center', color='skyblue')
        plt.title('Quantités vendues au cours des années')
        plt.xlabel('Année')
        plt.ylabel('Quantité totale vendue')
        plt.grid(axis='y')
        plt.xticks(quantities_per_year.index)
        plt.tight_layout()

        # Save the bar chart
        bar_chart_path = r'static/output2.png'
        plt.savefig(bar_chart_path)
        print(f"Bar chart saved to '{bar_chart_path}'.")

        # Close plots to avoid resource leakage
        plt.close('all')

        # Data visualization code ends here

        return redirect(url_for('dashboard'))

    return render_template('updatedata.html', banks=banks)

if __name__ == '__main__':
    app.run(debug=True)

