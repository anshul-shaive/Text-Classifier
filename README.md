Text Classifier
------------------


### Running Instructions:
 - clone the repo
 - cd into text_classifier
 - Run `pip install -r requirements.txt`


### Working:

 - There are two models: 
   - RNN model with GRU layer that works with tokenized text
   - Random Forest model that works with USE (Universal Sentence Encoder) Embeddings
 
   - [Notebooks](/text_classifier/src/notebooks) for both models
   - Python scripts can be used for both training and inference (by csv file or api)

 - RNN model:
   - To train the model: </br> `python -m src.models.RnnModel.train --train_data=data\\train_set.csv --model_dir=model`
   Pass the path to training data file in train_data and pass path to the folder in which to save models after training
   - To make predictions on entire test set: </br>
   `python -m src.models.RnnModel.inference` </br>
   This will create a test_labels.csv file in data directory. See below for inference through API
   
 - Random Forest Model:
   - To train the model: </br>
     `python -m src.models.RFModel.model` </br>
   First run will take longer as the USE model will be downloaded from tensorflow hub and cached subsequently
   - To make predictions on entire test set: </br>
    `python -m src.models.RFModel.inference` </br>
   This will create a test_labels_rf.csv file in data directory.

 - FLASK API
   - Run `python -m api_server.app` to start the server
   - Example API calls:
     - model_name can be random_forest or rnn
     - Send a POST request to `http://127.0.0.1:5000/text_classifier?model_name=random_forest`
             with JSON data in body as: </br>
              `{"text": [" mqs bugehpsw  housing   pcs"," pdscpm gb part of panel of chiller", 
" screwcmxxhni qty  pcs  of ce  partscomponents  accessories for the manufacturing"]}`
           - Response: </br>
          `{
          "prediction": [
              85389000,
              39269099,
              73181500
          ],
          "text": [
              " mqs bugehpsw  housing   pcs",
              " pdscpm gb part of panel of chiller",
              " screwcmxxhni qty  pcs  of ce  partscomponents  accessories for the manufacturing"
          ]
}`
           - Or send a get request like `http://127.0.0.1:5000/text_classifier?data={"text": [" mqs bugehpsw  housing   pcs"]}&model_name=rnn`
     