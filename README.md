# Customer Churn Prediction Project

This project aims to predict customer churn using machine learning techniques. The model is trained on a dataset containing customer information and their churn status. A Flask API is provided for making predictions based on new customer data.

## Project Structure

```
customer_churn_project/
├── data/
│   └── customer_churn.csv       # Dataset for customer churn (download manually)
├── model/
│   └── churn_artifacts.pkl      # Trained model and feature columns
├── notebooks/                   # Optional Jupyter notebooks for analysis
├── train.py                     # Script for training the model
├── app.py                       # Flask API for predictions
├── test_request.py              # Script to test the API
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

1. **Clone the repository** (if applicable):
   ```
   git clone <repository-url>
   cd customer_churn_project
   ```

2. **Download the dataset**:
   - Download the customer churn dataset and place it in the `data/` directory as `customer_churn.csv`.

3. **Install dependencies**:
   - Create a virtual environment (optional but recommended):
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
   - Install the required packages:
     ```
     pip install -r requirements.txt
     ```

## Usage

### Training the Model

To train the model, run the following command:
```
python train.py
```
This will preprocess the data, train an XGBoost classifier, and save the trained model along with the feature columns in `model/churn_artifacts.pkl`.

### Running the Flask API

To start the Flask API, run:
```
python app.py
```
The API will be available at `http://127.0.0.1:5000/`.

- **Check API Status**: Send a GET request to `/` to check if the API is running.
- **Make Predictions**: Send a POST request to `/predict` with customer details in JSON format.

### Testing the API

To test the API, run:
```
python test_request.py
```
This script will send a sample JSON request to the prediction endpoint and print the status code and JSON response.

## Notes

- Ensure that the preprocessing steps in `train.py` match those in `app.py` for consistent predictions.
- Modify the sample JSON in `test_request.py` to match the expected input format for predictions.

## License

This project is licensed under the MIT License.
