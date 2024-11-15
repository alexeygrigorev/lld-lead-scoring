# LLM-Driven Development for ML

This is a short course about using LLMs for assisting with
developing, training and deploying a machine learning model
for lead scoring.

Technologies we use in the course

* Claude.AI as the LLM-based chatbot
* Python 
* Github Codespaces
* PyData stack: Jupyter, Pandas, NumPy, Scikit-Learn
* Docker
* MLFlow


## Course content

Course plan:

* Identify a problem (Lead scoring)
* Generate a dataset
* Create code for training a model
* Automate model training
* Export the model to ML Flow
* Deploy the model
* Use the results


### 1. Course Introduction

* Course overview
* Lead scoring problem

Links:

* [Video](https://www.loom.com/share/73e013901bda47b8a4123b563cc0e38e)
* [Prompts](prompts/01-intro.md)
* [Slides](https://docs.google.com/presentation/d/19XAVPOAOx00NcvFcUSIWBatV53Nr2wpt-6AmJpGty1U/edit)


### 2. Dataset generation

* Problem overview
* Using LLM to generate prompts
* Dataset overview
* Generating data
* Prepare the enviroment (codespaces)

Links:

* [Video](https://www.loom.com/share/4beba860d8c24e4c8c1485bd4f79cf44)
* [Prompts](prompts/02-data.md)
* [Notebook](analytics/users.ipynb)


To open the Terminal in VS Code, use ``` Ctrl+` ```

Installing libraries:

```bash
pip install pandas numpy scikit-learn jupyter
```

Running Jupyter:

```bash
jupyter notebook
```


### 3. Training the first model

* Training Logistic Regression
* One-Hot Encoding


Links:

* [Video](https://www.loom.com/share/74ecce75606b463ea4947661b13ce46d)
* [Prompts](prompts/03-train.md)
* [Notebook](analytics/train-01.ipynb)


### 4. Feature engineering 

* Extracting features from logs

Links:

* [Video](https://www.loom.com/share/2e48598f016d4add81952be7a13f1e97)
* [Prompts](prompts/04-feature-engineering.md)
* [Notebook](analytics/train-02.ipynb)

### 5. Experiment tracking

* Experiment tracking with MLFlow


Links:

* [Video](https://www.loom.com/share/a546793dc628431f948135360e0edd19)
* [Prompts](prompts/05-mlflow.md)
* [Notebook](analytics/train-03-mlflow.ipynb)

Installing mlflow

```bash
pip install mlflow
```

Running MLFlow:

```bash
mlflow ui

mlflow server \
   --backend-store-uri sqlite:///mlflow.db \
   --default-artifact-root mlruns \
   --host 0.0.0.0 \
   --port 5000
```

### 6. Scikit-Learn Pipelines (optional)

* Code cleaning 
* Putting the code into a pipeline


Links:

* [Video](https://www.loom.com/share/5a9c4278cdfb47178579ef12159ef1ff)
* [Prompts](prompts/06-pipelines.md)
* [Notebook](analytics/train-04-pipelines.ipynb)


### 7. Training tree-based models: Decision Tree and Random Forest

* Training decision tree
* Training random forest

Links:

* [Video](https://www.loom.com/share/d7bebf90d4b94d2cb0756b03d82ef348)
* [Prompts]()
* [Notebook](analytics/train-05-trees.ipynb)

### 8. XGBoost