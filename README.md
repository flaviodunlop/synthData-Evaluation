
# synthData-Evaluation
A standardised framework for evaluating the utility and privacy of synthetic tabular data.

# Run Evaluation Framework
## Run with Docker
A docker image of the complete framework including the streamlit app is available:
`docker pull flaviodunlop/synthdata-eval:latest`

To start the container:
`docker run -p 8501:8501 flaviodunlop/synthdata-eval`
and access the app at: http://localhost:8501

If port 8501 is already occupied on your local machine, you can map the container port to a different local port. For example:
`docker run -p 8502:8501 flaviodunlop/synthdata-eval`
and access the app at: http://localhost:8502

## Run without Docker
Use python = 3.9

Requirements:
```
streamlit==1.44.1
numpy==2.0.1
pandas==2.2.3 
scikit-learn==1.6.1
xgboost==2.1.4
```

Or install dependencies with requirements.txt-File (optional):
`pip install -r requirements.txt`

And run the app.py file in the EvaluationFramwork-Folder:
`streamlit run app.py`

# Benchmark Datasets
The data folder contains cross-validation in- and out-sets of the Adult and News datasets from the UCI Machine Learning Repository.
The in-sets can be used to train a GAN. Later, the in-sets, out-sets, and the generated synthetic sets are fed into the evaluation framework to assess utility and privacy.