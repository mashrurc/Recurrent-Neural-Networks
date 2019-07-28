# Recurrent-Neural-Networks
Demo of an RNN trained on NASDAQ:GOOG.
Trained on GOOG pricing history from 2012-2016.
Predicts full month trend for January 2017.
Can be applied to other stocks with minimal tuning and .csv file

Todo:
- try different timesteps, currently uses 3 months of previous data to make next days prediction each iteration.
- trying different number of neurons per layer
- adding higher dimensionalitt thorough layerization and/or 3rd dimension indicators.
- changing batch size, learning rate, optimizers and loss function
- deploying program with newer data (up to June 2019) and testing performance
- deploying on smaller priced stocks with less volatility (FB, Sprint, Tesla)
- changing epochs (currently 100 iterations until full train)
