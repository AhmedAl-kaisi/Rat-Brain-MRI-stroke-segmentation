You will need to change the path specified in CNNv6.py to whatever the path where the images/masks are stored. Rename the file the model is stored to (currently v1hemmorage.keras) and ensure it
matches the one being used in the front end (currently best_model3.keras)
The actual model follows a U-Net structure with named forward and upsampling blocks. The model can easily be altered by adding/removing these forward and upsampling blocks however ensure you correctly change the first parameter being passed in/ensure it follows the same structure it follows now.
