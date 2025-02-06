# F1-Overtake-Predictor
Neural Network predicting the number of overtakes during an F1 race using past data from Ergast API

 1 - Collecting Data

 All data was collected from the Ergast API through FastF1 for python. I chose to use all races from the 2014 season onwards as I felt races before the hybrid era may not align and cause inconsistencies with the model. For each race, the following features were extracted and appended to the dataframe in the following order:

   - Nb of Overtakes (Int)
   - Country (String)
   - Nb of Yellow Flags (Int)
   - Nb of Red Flags (Int)
   - Nb of Safety Cars (Int)
   - Nb of Virtual Safety Cars (Int)
   - Rain (Bool)

The following 20 features were composed as follows: for each driver (1 to 20) we determine a score which is "starting position - end position". For example if you qualify 5th and finish 3rd, you obtain a score of +2. I chose this scoring system as I believed the model could pick up trends such as the whoever qualifies first rarely ends on pole which therefore impacts the total number of overtakes. Each score was them divided by 20 normalizing them between -1 and 1. I then manually cleaned the data filling in a couple missing race results. Each dataframe is finally saved as well as a copy of it under its year name.

2 - Preparing Data

All csv files were then appended a prepared to be fed into the model for training. This involved mapping each country to a valid Race ID.
