document.addEventListener('DOMContentLoaded', (event) => {
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent the default form submission

        const formData = new FormData(form);
        const data = {};

        // Convert form data into JSON
        formData.forEach((value, key) => {
            const parsedValue = parseFloat(value);
            // Only add the value if it's a valid number
            if (!isNaN(parsedValue)) {
                data[key] = parsedValue;
            } else {
                console.warn(`Invalid input for ${key}: ${value}`);
            }
        });

        console.log('Data being sent:', data);  // Log data for debugging

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            console.log('Response from backend:', result);  // Log response from backend
            resultDiv.textContent = `Diagnosis: ${result.diagnosis} (Probability: ${(result.probability * 100).toFixed(2)}%)`;
            resultDiv.classList.remove('error');
        } catch (error) {
            console.error("Error during fetch:", error);  // Log error for debugging
            resultDiv.textContent = 'Error: Could not fetch prediction. Is the backend running?';
            resultDiv.classList.add('error');
        }
    });
});
