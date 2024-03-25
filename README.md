Building out original ECO EVO SIM idea in Python before translating to C++


To Do:

1) Revert back to using a dictionary to keep track of main data
   a) Need to find most efficient way
   b) Periodically store updates in DB (after each turn) but don't always rely on pulling from DB to refresh operational data

2) Start setting up system for NN brain to interact with the genes
   a) Need to build out environment first
   b) Have movement within the environment associated with the genes somehow
