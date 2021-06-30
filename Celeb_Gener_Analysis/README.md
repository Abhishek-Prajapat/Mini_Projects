# Celeb-Gender-Prediction

The dataset has been taken from - https://www.kaggle.com/jessicali9530/celeba-dataset

The Basic EDA of the dataset is in **notebook/EDA.ipynb**.

Due to less processing power we have trained the model on about 10,000 images in compare to 2,00,000+ in the whole dataset.

To train the model you need to execute the **run.sh** file and if you are training the model on **Colab** then execute the cell in the **Colab_run.ipynb**.

To modify the training you can edit the run.sh file.

The **requirements.txt** file contains the python libraries need for training of the model and the **requirements_inferance.txt** file contains the python libraries needed during the prediction.

To add more models you need to edit the **model.py** and **model_dispatcher.py** files.
