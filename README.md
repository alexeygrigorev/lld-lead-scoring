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


### 4. Feature engineering 

* Extracting features from logs

Links:

* [Video](https://www.loom.com/share/2e48598f016d4add81952be7a13f1e97)
* [Prompts](prompts/04-feature-engineering.md)

### 5. Training more models

* Experiment tracking with MLFlow
* Training random forest and xgboost