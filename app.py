import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template
import pandas as pd
import re
import os
import shutil

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('fine_tuned_lstm_model.keras')

# Functions
def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def predict_sentiment(review):
    review_cleaned = remove_emojis(review)
    review_cleaned = re.sub(r'[^a-zA-Z0-9\s\.\,\-]', '', review_cleaned)
    review_cleaned = review_cleaned.strip()
    sequence = tokenizer.texts_to_sequences([review_cleaned])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence, verbose=0)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Initialize Flask app
app = Flask(__name__, template_folder='temp')
app.static_folder = 'static'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Route for home page (CSV upload)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                if 'Comments' not in df.columns:
                    return render_template('index.html', error="CSV must have a 'Comments' column")
                sentiments = df['Comments'].apply(predict_sentiment)
                total_comments = len(sentiments)
                positive_count = sum(1 for s in sentiments if s == 'positive')
                negative_count = total_comments - positive_count
                positive_percentage = (positive_count / total_comments) * 100
                recommendation = "Stock the graphics cards" if positive_percentage > 60 else "Do not stock the graphics cards"
                result = {
                    'total_comments': total_comments,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'positive_percentage': round(positive_percentage, 2),
                    'recommendation': recommendation
                }
                return render_template('result_doughnut.html', result=result)
            except Exception as e:
                return render_template('index.html', error=f"Error processing file: {str(e)}")
    return render_template('index.html')

# Route for single text prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        review = request.form.get('review')
        if not review:
            return render_template('predict.html', error="Please enter a review")
        sentiment = predict_sentiment(review)
        return render_template('predict.html', review=review, sentiment=sentiment)
    return render_template('predict.html')

# Clear and recreate temp folder
if os.path.exists('temp'):
    shutil.rmtree('temp')
os.makedirs('temp')
if not os.path.exists('static'):
    os.makedirs('static')

# Write your templates (index.html, result_doughnut.html, predict.html) here
# [Copy the HTML from your previous version here]
with open('temp/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graphics Card Sentiment Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.3)), url('/static/background.jpg') no-repeat center/cover fixed;
            color: #333;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            max-width: 800px;
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            text-align: center;
            animation: fadeIn 1s ease-in;
        }
        h1 {
            color: #1a3c5e;
            font-size: 2.5em;
            margin-bottom: 10px;
            animation: slideIn 0.8s ease-out;
        }
        .tagline {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        input[type="file"] {
            font-size: 1.1em;
            padding: 10px;
            border: 2px dashed #3498db;
            border-radius: 8px;
            width: 100%;
            max-width: 400px;
        }
        input[type="submit"] {
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.2em;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        input[type="submit"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        .error {
            color: #e74c3c;
            font-weight: 600;
            margin-top: 15px;
        }
        .nav-link {
            display: block;
            margin-top: 25px;
            color: #3498db;
            text-decoration: none;
            font-size: 1.1em;
            transition: color 0.3s;
        }
        .nav-link:hover {
            color: #1a3c5e;
            text-decoration: underline;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Graphics Card Review Analyzer</h1>
        <p class="tagline">Analyze customer sentiments to make smart stocking decisions</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv">
            <input type="submit" value="Analyze Now">
        </form>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        <a href="/predict" class="nav-link">Analyze a Single Review</a>
    </div>
</body>
</html>''')  # Replace with your actual index.html


with open('temp/result_doughnut.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.3)), url('/static/background.jpg') no-repeat center/cover fixed;
            color: #333;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            max-width: 900px;
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            animation: fadeIn 1s ease-in;
        }
        h1 {
            color: #1a3c5e;
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 20px;
            animation: slideIn 0.8s ease-out;
        }
        .result {
            background: #f7f9fc;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }
        .stats-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            flex: 1;
            min-width: 150px;
            text-align: center;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        }
        .stat-card p {
            margin: 5px 0;
            font-size: 1em; /* Reduced from 1.1em */
        }
        .recommendation {
            font-weight: 600;
            color: #27ae60;
            font-size: 1.4em;
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: rgba(39, 174, 96, 0.1);
            border-radius: 8px;
        }
        .back-link {
            display: block;
            text-align: center;
            margin-top: 30px;
            color: #3498db;
            text-decoration: none;
            font-size: 1.2em;
            transition: color 0.3s;
        }
        .back-link:hover {
            color: #1a3c5e;
            text-decoration: underline;
        }
        #sentimentChart {
            max-width: 400px;
            margin: 0 auto 20px auto;
            filter: drop-shadow(0 5px 15px rgba(0, 0, 0, 0.2));
        }
        .chart-buttons {
            text-align: center;
            margin: 20px 0;
        }
        .chart-button {
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 1em;
            margin: 0 10px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .chart-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        .chart-button.active {
            background: linear-gradient(90deg, #1a3c5e, #2c3e50);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Analysis Results</h1>
        <div class="result">
            <div class="stats-row">
                <div class="stat-card">
                    <p>Total Comments: {{ result.total_comments }}</p>
                </div>
                <div class="stat-card">
                    <p>Positive: {{ result.positive_count }}</p>
                </div>
                <div class="stat-card">
                    <p>Negative: {{ result.negative_count }}</p>
                </div>
                <div class="stat-card">
                    <p>Positive %: {{ result.positive_percentage }}%</p>
                </div>
            </div>
            <canvas id="sentimentChart"></canvas>
            <div class="chart-buttons">
                <button class="chart-button active" id="doughnutBtn">Doughnut Chart</button>
                <button class="chart-button" id="barBtn">Bar Chart</button>
            </div>
            <p class="recommendation">Recommendation: {{ result.recommendation }}</p>
        </div>
        <a href="/" class="back-link">Analyze Another File</a>
    </div>
    <script>
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        let sentimentChart;

        const data = {
            labels: ['Positive', 'Negative'],
            datasets: [{
                label: 'Number of Comments',
                data: [{{ result.positive_count }}, {{ result.negative_count }}],
                backgroundColor: ['#2ecc71', '#e74c3c'],
                borderColor: ['#27ae60', '#c0392b'],
                borderWidth: 2,
                hoverOffset: 10
            }]
        };

        const doughnutConfig = {
            type: 'doughnut',
            data: data,
            options: {
                plugins: {
                    legend: { position: 'top', labels: { font: { size: 14, family: 'Poppins' } } },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.raw;
                            }
                        }
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true
                },
                cutout: '60%'
            }
        };

        const barConfig = {
            type: 'bar',
            data: data,
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Count', font: { size: 14, family: 'Poppins' } }
                    },
                    x: {
                        title: { display: true, text: 'Sentiment', font: { size: 14, family: 'Poppins' } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.raw;
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutBounce'
                }
            }
        };

        // Start with doughnut chart
        sentimentChart = new Chart(ctx, doughnutConfig);

        // Switch chart function
        function updateChart(type) {
            sentimentChart.destroy();
            if (type === 'doughnut') {
                sentimentChart = new Chart(ctx, doughnutConfig);
                document.getElementById('doughnutBtn').classList.add('active');
                document.getElementById('barBtn').classList.remove('active');
            } else {
                sentimentChart = new Chart(ctx, barConfig);
                document.getElementById('barBtn').classList.add('active');
                document.getElementById('doughnutBtn').classList.remove('active');
            }
        }

        // Button event listeners
        document.getElementById('doughnutBtn').addEventListener('click', () => updateChart('doughnut'));
        document.getElementById('barBtn').addEventListener('click', () => updateChart('bar'));
    </script>
</body>
</html>''') # Replace with your actual result_doughnut.html


with open('temp/predict.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict Sentiment</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.3)), url('/static/background.jpg') no-repeat center/cover fixed;
            color: #333;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            max-width: 800px;
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            text-align: center;
            animation: fadeIn 1s ease-in;
        }
        h1 {
            color: #1a3c5e;
            font-size: 2.5em;
            margin-bottom: 10px;
            animation: slideIn 0.8s ease-out;
        }
        p {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        textarea {
            width: 100%;
            max-width: 500px;
            height: 120px;
            padding: 15px;
            font-size: 1.1em;
            border: 2px solid #ddd;
            border-radius: 8px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        input[type="submit"] {
            background: linear-gradient(90deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.2em;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        input[type="submit"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        .error {
            color: #e74c3c;
            font-weight: 600;
            margin-top: 15px;
        }
        .result {
            margin-top: 25px;
            font-size: 1.3em;
        }
        .positive { color: #27ae60; font-weight: 600; }
        .negative { color: #e74c3c; font-weight: 600; }
        .back-link {
            display: block;
            margin-top: 30px;
            color: #3498db;
            text-decoration: none;
            font-size: 1.2em;
            transition: color 0.3s;
        }
        .back-link:hover {
            color: #1a3c5e;
            text-decoration: underline;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predict Sentiment</h1>
        <p>Enter a graphics card review to analyze its sentiment</p>
        <form method="post">
            <textarea name="review" placeholder="Write your review here..."></textarea>
            <input type="submit" value="Predict Sentiment">
        </form>
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        {% if review and sentiment %}
            <div class="result">
                <p>Your review: "{{ review }}"</p>
                <p>Sentiment: <span class="{{ sentiment }}">{{ sentiment|capitalize }}</span></p>
            </div>
        {% endif %}
        <a href="/" class="back-link">Back to CSV Upload</a>
    </div>
</body>
</html>''')  # Replace with your actual predict.html

# For local testing only; Render uses gunicorn
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
