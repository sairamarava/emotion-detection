document.getElementById("emotion-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const description = document.getElementById("description").value;
    const resultElement = document.getElementById("result");

    try {
        const response = await fetch("https://<YOUR_BACKEND_URL>/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ description }),
        });

        const data = await response.json();
        resultElement.innerText = `Detected Emotion: ${data.prediction}`;
    } catch (error) {
        resultElement.innerText = "Error detecting emotion. Please try again.";
    }
});
