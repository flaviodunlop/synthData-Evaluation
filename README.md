# synthData-Evaluation
A standardised framework for evaluating the utility and privacy of synthetic tabular data.

# Evaluation Framework


A docker image of the complete framework including the streamlit app is available:
`docker pull flaviodunlop/synthdata-eval:latest`

To start the container:
`docker run -p 8501:8501 flaviodunlop/synthdata-eval`
and access the app at: http://localhost:8501

If port 8501 is already occupied on your local machine, you can map the container port to a different local port. For example:
`docker run -p 8502:8501 flaviodunlop/synthdata-eval`
and access the app at: http://localhost:8502

# Benchmark Datasets
The data folder contains cross-validation in- and out-sets of the Adult and News datasets from the UCI Machine Learning Repository.
The in-sets can be used to train a GAN. Later, the in-sets, out-sets, and the generated synthetic sets are fed into the evaluation framework to assess utility and privacy.