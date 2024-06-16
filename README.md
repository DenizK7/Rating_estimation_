
# Review Rating Prediction API

This project provides a REST API for predicting review ratings based on review text using a BERT model and a Keras neural network.

## Requirements

- Python 3.8
- Virtual environment (`venv`)

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DenizK7/Rating_Estimation.git
   cd <repository_directory>
2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv ml_env
    source ml_env/bin/activate  # On Windows, use `ml_env\Scripts\activate
3. **Install dependencies:**:
    ```bash
    pip install -r requirements.txt
4. **Run the API:**:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
**Usage**:

To use the API, send a POST request to /predict with the following JSON payload:

```
{
  "review_text": "The product is excellent!",
  "summary": "Great product",
  "helpful_ratio": 0.8
}
```
**Example using Postman(Recommended)**:

 ```http://localhost:8000/predict```
 then select body. After that select raw and json,
 ```
 {
  "review_text": "I love reading things involving our 16th president.I got to learn and enjoy reading this & would suggest it to others.It gets an A+ from this reader",
  "summary": "Wonderful stories!",
  "helpful_ratio": 0.8
}
```
**Example Response**:
```
{
  "predicted_rating": 4.5
}
```
**Testing**:

To run unit tests for the API:


```
python -m unittest discover -s tests
```

## Docker Instructions
   
**Build the Docker Image**
```
docker build -t my-fastapi-app .
```

**Run the Docker Container**
```
docker run -d -p 8000:80 my-fastapi-app
```
## NOTES
- Ensure that the rating_predictor.pkl model file is present in the root directory of the project.
- The API is built using FastAPI and can be extended with additional endpoints as needed.
