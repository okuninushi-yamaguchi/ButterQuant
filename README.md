# ARIMA-GARCH Butterfly Option Analyzer

This application consists of two parts: a Python Flask backend for complex financial calculations and a React frontend for visualization.

## 1. Backend Setup (Python)

You need Python installed on your machine.

1.  Navigate to the `backend` folder (you need to create this manually or move the files `app.py` and `requirements.txt` into a folder named `backend`).
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the server:
    ```bash
    python app.py
    ```
    The server will start at `http://localhost:5000`.

## 2. Frontend Setup (React)

1.  Install Node.js dependencies:
    ```bash
    npm install
    ```
2.  Run the development server:
    ```bash
    npm start
    ```
3.  Open your browser at the address shown (usually `http://localhost:8080` or similar).

## Troubleshooting

-   **CORS Errors:** The backend is configured with `CORS(app)`, so it should accept requests from the frontend. Ensure the backend is running on port 5000.
-   **Data Errors:** If the analysis fails, ensure the stock ticker is valid and Yahoo Finance data is accessible from your network.
