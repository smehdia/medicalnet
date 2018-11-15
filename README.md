# medicalnet
Diabetes Classification using Retina Images


How To RUN:

1) Go to 'Preprocess_and_Save_Images_as_numpy_array' and then modify CSV_PATH and IMAGES_PATH in 'create_data.py' according to your dataset directory

2) Run 'create_data.py' using below command:
  python create_data.py
 
3) (optional) If you want to use pretrained GANs skip this step, otherwise go to DCGAN directory and open 'dcgan.py', then change CLASS according to the class which you want to be learned. then run 'dcgan.py' using below command:
  python dcgan.py
  after training the model, copy it in the relevant class
 
4) Go to the 'Classifier' directory and run 'create_data' using below command:
  python create_data.py

5) Run 'train.py' using below command:
  python train.py
 
  ** YOU COULD CHANGE NUMBER OF GPUs IN 'train.py'. for this, you should change 'NUM_GPUs' in 'train.py'
