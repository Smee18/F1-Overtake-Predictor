# F1-Overtake-Predictor

Neural Network predicting the number of overtakes during an F1 race using past data from Ergast API. 

As a passionate F1 fan, following the sport for over 10 years with my father, I have been working on this personal project to apply my Machine Learning skills, leveraging real data to simulate race outcomes. All code and data is linked and should all run given the correct libraries being installed.

# 1 - Collecting Data

 All data was collected from the Ergast API through FastF1 for python. I had to use all races from the 2018 season onwards as the API could not fetch lap, driver and weather data for races before then. For each race, the following features were extracted and appended to the dataframe in the following order:

   - Nb of Overtakes (Int) - Target variable
   - Country (String)
   - Nb of Yellow Flags (Int)
   - Nb of Red Flags (Int)
   - Nb of Safety Cars (Int)
   - Nb of Virtual Safety Cars (Int)
   - Rain (Bool)

The following 20 features were composed as follows: for each driver (1 to 20) we determine a score which is "starting position - end position". For example if you qualify 5th and finish 3rd, you obtain a score of +2. I chose this scoring system as I believed the model could pick up trends such as the whoever qualifies first rarely ends winning which therefore impacts the total number of overtakes. Each score was them divided by 20 normalizing them between -1 and 1. I then manually cleaned the data filling in a couple missing race results. Each dataframe is finally saved as well as a copy of it under its year name. All csv files were then appended a prepared to be fed into the model for training. This involved mapping each country to a valid Race ID.

Here are the first 5 elements of the feature data: 

<img width="941" alt="Screenshot 2025-02-06 124747" src="https://github.com/user-attachments/assets/763ef4ae-8f3d-4f8e-a129-418b33042a9f" />

# 2 - Data and Network Setup

The neural network was created using the PyTorch library as a subclass of nn.Module so I could manually change the architecture to fine tune it.

The csv file is imported, and split into the features and target data. We allocate a 80/20 split for training and test data. Data is them normalized and standardized to ensure consistency. Finally, it is converted to tensors for the model to work with.

The learning rate is set to 0.1 which L2 regularization of 0.01. I also set up a scheduler to halve the learning rate every 100 epochs to improve convergence.

In order to maximize the model's score I tried various network architecture to find out which worked best. I experimented with various numbers of hidden layers, normalization layers and dropout layers. Leaky Rely was used as an activation layer to avoid vanishing gradient. Weights were therefore initialized using He initialization as it works with ReLu activation functions. 

# 3 - Training 

The model was trained over 1000 epochs which batch sizes of 16. Mean Absolute Error was used as to measure performance. The model was trained with and without certains layers to experiment different architectures.

# 4 - Evaluation

Below are the scores for 4 different model architectures each trained on the same data, in the same order, with the same learning rate and scheduler, and for the same number of epochs. The results were as follow:

![tt](https://github.com/user-attachments/assets/492416d4-e42e-4d20-966f-bb792d7fd11c)
![tf](https://github.com/user-attachments/assets/c56300aa-d100-4520-8a0c-c3f281859b27)
![ft](https://github.com/user-attachments/assets/bffcd538-8fcd-4ec1-b62b-3ee07ed2223f)
![ff](https://github.com/user-attachments/assets/1d022e24-bbdb-4dab-8c84-c5b059db26a1)

Finally, here is a loss graph for the optimal model: 

![graph](https://github.com/user-attachments/assets/a6791a79-d12a-4364-9fdb-9c299378548d)

# 5 - Conclusion and further work

The main conclusion we can draw from experimenting with the models is that more complex architectures does not always mean better models. We notice that the architecture that yielded the best result had only one hidden layer and neither Dropout nor Layer Normalization. It is important to point out that the dataset is very small (only around 150 entries). This project could be improved by finding a way to expand the dataset size to improve generability. It could also be wise to try ensemble methods such as a random forest and compare performance. 

