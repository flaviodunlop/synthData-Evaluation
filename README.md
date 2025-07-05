# synthData-Evaluation

A standardized framework for evaluating the **utility** and **privacy** of synthetic tabular data.

---

## Running the Evaluation Framework

### Using Docker (Recommended)

A pre-built Docker image containing the complete framework, including the Streamlit app, is available:

```bash
docker pull flaviodunlop/synthdata-eval:latest
```

To start the container:

```bash
docker run -p 8501:8501 flaviodunlop/synthdata-eval
```

You can then access the app at: [http://localhost:8501](http://localhost:8501)

If port 8501 is already in use on your local machine, you can map the container port to a different local port. For example:

```bash
docker run -p 8502:8501 flaviodunlop/synthdata-eval
```

Access the app at: [http://localhost:8502](http://localhost:8502)

---

### Running Without Docker

Ensure Python **3.9** is installed.

**Required packages:**

```text
streamlit==1.44.1
numpy==2.0.1
pandas==2.2.3 
scikit-learn==1.6.1
xgboost==2.1.4
```

To install dependencies using the `requirements.txt` file (optional):

```bash
pip install -r requirements.txt
```

Then run the app located in the `EvaluationFramework` directory:

```bash
streamlit run app.py
```

---

## Benchmark Datasets

The `data` folder contains cross-validation **in-sets** and **out-sets** of the **Adult** and **News** datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

Use the **in-sets** to train a generative model (e.g., a GAN). Once synthetic data is generated, use the framework to evaluate it against the **in-sets** and **out-sets** to asses the **utility** and  the **privacy**.

