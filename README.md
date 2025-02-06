# F1-Overtake-Predictor
Neural Network predicting the number of overtakes during an F1 race using past data from Ergast API

# 1 - Collecting Data

 All data was collected from the Ergast API through FastF1 for python. I had to use all races from the 2018 season onwards as the API could not fetch lap, driver and weather data for races before then. For each race, the following features were extracted and appended to the dataframe in the following order:

   - Nb of Overtakes (Int)
   - Country (String)
   - Nb of Yellow Flags (Int)
   - Nb of Red Flags (Int)
   - Nb of Safety Cars (Int)
   - Nb of Virtual Safety Cars (Int)
   - Rain (Bool)

The following 20 features were composed as follows: for each driver (1 to 20) we determine a score which is "starting position - end position". For example if you qualify 5th and finish 3rd, you obtain a score of +2. I chose this scoring system as I believed the model could pick up trends such as the whoever qualifies first rarely ends on pole which therefore impacts the total number of overtakes. Each score was them divided by 20 normalizing them between -1 and 1. I then manually cleaned the data filling in a couple missing race results. Each dataframe is finally saved as well as a copy of it under its year name.

# 2 - Preparing Data

All csv files were then appended a prepared to be fed into the model for training. This involved mapping each country to a valid Race ID.

# 3 - Data and Network Setup

The neural network was created using the PyTorch library as a subclass of nn.Module so I could manually change the architecture to fine tune it.

The csv file is imported, and split into the features and target data. We allocate a 80/20 split for training and test data. Data is them normalized and standardized to ensure consistency. Finally, it is converted to tensors for the model to work with.

The learning rate is set to 0.1 which L2 regularization of 0.01. I also set up a scheduler to decrease the learning rate by half every 100 epochs to improve convergence.

In order to maximize the model's score I tried various network architecture to find out which worked best. I experimented with various numbers of hidden layers, normalization layers and dropout layers. Leaky Rely was used as an activation layer to avoid vanishing gradient. Weights were therefore initialized using He initialization as it works with ReLu activation functions. 

# 4 - Training 

The model was trained over 1000 epochs which batch sizes of 16. Mean Absolute Error was used as to measure performance. The model was trained with and without certains layers to experiment different architectures.

# - Evaluation

Below are the scores for 4 different model architectures each trained on the same data, in the same order, with the same learning rate and scheduler, and for the same number of epoch. The results were as follow:

![tt](https://github.com/user-attachments/assets/492416d4-e42e-4d20-966f-bb792d7fd11c)
![tf](https://github.com/user-attachments/assets/c56300aa-d100-4520-8a0c-c3f281859b27)
![ft](https://github.com/user-attachments/assets/bffcd538-8fcd-4ec1-b62b-3ee07ed2223f)
![ff](https://github.com/user-attachments/assets/1d022e24-bbdb-4dab-8c84-c5b059db26a1)

