# FCELoss-Pytorch
Implementation of the Fair Channel Enhancement Loss in Pytorch

## Files
`main.py`: Contains the main code to peform the training and testing of the model.

`fce_loss.py`: Contains code that computes the FCE loss.

`dataset.py`: Contains code related to processing the SD dataset.

`augmentations.py`: Contains code used to apply image augmentations to the image samples.

`model.py`: Contains code that defines the modified backbone model. 

`train_py`: Contains code to train the model.

`test_py`: Contains code to test the model.

`utils_py`: Contains additional helpy functions

## Usage
To use the fair channel enhancement loss separately, import the `fce` function from the **fce_loss.py** file. Keep in mind the required parameters to be passed. The model must also be modified to output the extracted features separately. 

