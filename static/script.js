document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('model');

    fetch('/models')
        .then(response => response.json())
        .then(models => {
            if (models.length === 0) {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No models found';
                modelSelect.appendChild(option);
            } else {
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'Error loading models';
            modelSelect.appendChild(option);
        });
});

document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const bloodGlucose = document.getElementById('blood-glucose').value;
    const meal = document.getElementById('meal').value;
    const model = document.getElementById('model').value;
    const resultDiv = document.getElementById('result');

    if (!model) {
        resultDiv.innerHTML = 'Please select a model.';
        return;
    }

    resultDiv.innerHTML = 'Predicting...';

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            blood_glucose: bloodGlucose,
            meal: meal,
            model_name: model
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = 'Error: ' + data.error;
        } else {
            const prediction = parseFloat(data.prediction[0]).toFixed(2);
            resultDiv.innerHTML = 'Predicted Insulin Dosage: ' + prediction + ' Unit';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.innerHTML = 'An error occurred during prediction.';
    });
});
