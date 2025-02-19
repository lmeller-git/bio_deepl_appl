# Tutorial 2: Convolutional Neural Networks
  

  
In this tutorial we will focus on Convolutional Neural Networks (CNNs).  
  
We will train a model to label Brain MRI scans as either healthy or containing a tumor. 
  
We will use [Pytorch](https://pytorch.org/tutorials/recipes/recipes_index.html) and a [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/levels/core_skills.html) to build and train our model.  
  
The notebook will give you a framework with some parts of the code left blank. Fill in the missing code to make it work. 
  
It will be helpful to look up Pytorch or Lightning commands on the go. The packages usually offer easy-to-use methods for everything Deep Learning related. 
  

  
Before we start, let's explore how CNNs work in this [CNN visualizer](https://adamharley.com/nn_vis/cnn/3d.html)  
## Step 0: Imports
  

  
Let's import all the Python modules that we need.  
## Step 1: Loading the data
  

  
Download the dataset from: https://polybox.ethz.ch/index.php/s/cXXOTIJowCJqMbz
  

  
The dataset contains images of brain MRI scans. Some of them are from healthy patients, others from patients with brain tumors.  
  
We will train a model that can classify the images correctly.
  

  
Unzip the downloaded folder. 
  
Then, add the correct path in the cell below, pointing to the data.   
How large is our dataset?
  
Plot the class distribution below:  

  
Now, that we have loaded the labels, we will load the images.  
  
As we have seen above the images can have different file types and dimensions.   
  
Next we will load the data into datasets that we can use for training. 
  
To simplify training the models, we will transform the pictures, to the same size and normalize the data.  
Let's take a look at some of the images we have loaded.  
Under the hood the images are just vectors though.  
  
In the dataset each item is saved together with it's label. 
  
The images in the dataset all have a size of 256x256 and 3 color channels (RBG).   
The above vector/tensor encodes the picture below.  
## Step 2: Creating a CNN architecture
  

  
Here is a recap of how convolutions work and CNNs work:
  
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53   
### Helper function
  

  
This function computes the new dimensions of our data after convolution and max pooling.  
  
It will be helpful later when we build the model.  
**Can you explain the output above? How does the size change after a convolution and why?**
  

  
Your answer:  
### CNN Model
  

  
We can now build our Convolutional Neural Network. 
  
It will have two convolutional layers and two fully connected layers.  
  
You will be able to mostly use Pytorch methods to fill in the blanks.   
The code above is defining a class. This will allow us to create objects of that class. 
  

  
**What does ``self.`` do in the code? What is it good for in a class?**
  

  
Your answer:
  

  
Let's try an example and see how the model works.  
**Explain the shapes of the batch and the output shown above.**
  

  
Your answer:
  

  
**How do you interpret the prediction for the first image?**
  

  
Your answer:  
### Train and validation with Lightning
  

  
Now that we have created our model architecture, we will wrap a Lightning module around it. 
  
This will make the training procedure much easier.  
  
Instead of programming the whole training loops ourselves, we will define how one step should be handled at training, validation and testing.  
  
We only need to define how to retreive data from the batch, how to pass it through our model and how/when to compute the loss. 
  
The rest will be all handled by Lightning.  
  
Make sure to use the Lightning docs and Google, to find the right methods.    
### Visualization
  
To visualize the training procedure, we will use Tensorboard.  
  
In the code above we can see some values being logged. Tensorboard will display these values in nice graphs for us to follow our learning curves.  
## Step 3. Training and evaluating 
  

  
Now that everything is ready, we can start training the model.  
  
Make sure to follow its performance on Tensorboard.   
If you get the message ``Error displaying widget`` stop jupyter lab, run ``conda uninstall ipywidgets`` in the terminal and start jupyter lab again.
  

  
Look at the learning curves in tensorboard. (You might need to click the refresh button on the website). 
  
Answer the following questions below. Write the answers in this cell.
  

  
1. **How many steps are there in one epoch? How do you compute it?**
  

  
    Your answer:
  

  
2. **What is the difference between the metrics per step and per epoch?**
  

  
    Your answer:
  

  
3. **Which metrics/graphs can help you understand whether your model is learning something useful from the data?**
  

  
    Your answer
  

  
4. **How well did your model train? Would you improve something? Explain your answer.**
  

  
    Your answer:
  

  
5. **How could you see from the graphs if your model is overfitting?**
  

  
    Your answer:  
### Validate and visualize
  

  
Let us evaluate the model now. As we might still make changes, and tune parameters, we should not use the test set, yet. 
  
The test set is only for the final evaluation and should never be looked at before to ensure unbiased models.  
We will visualize our predictions in a confusion matrix to get a feeling of how well the model performs in specific cases.   
**How does the information we get from the confusion matrix compare to what we can learn from the training curves?**
  

  
Your answer:  
In the U.S. 21.97 of 100,000 people are diagnosed with brain tumors.  
  
Assume a doctor uses our model to screen 100,000 persons from the U.S. 
  

  
**Based on the computed values above, how many healthy people do we expect to be wrongly diagnosed with brain cancer?**
  

  
Your answer:  
## Step 4: Improving
  

  
### Finetuning training parameters
  

  
The training procedure could use some improvements.  
  
Adjust the number of epochs, batch size and learning rate and rerun the model.  
  
Analyze how performance changes.    
**How does the learning rate influence training performance?**
  

  
Your answer:
  

  

  

  

  
Let's use the original learning rate of 1e-3 again. 
  
Now we will change the batch size in the dataloaders  
**How does the batch size influence the model performance?**
  

  
Your answer:
  

  
Now let's train for more epochs.  
**How does the model perform? How do more epochs influence performance and how many epochs are enough?**
  

  
Your answer:  
### Model improvements (optional)
  

  
Simply finding the best training parameters improves the performance to some degree.  
  
Especially in more complex problems and with larger datasets the architecture and the amount and size of the layers also matter. 
  
Now you can experiment with the CNN architecture. 
  
Use the code from a bove and create a deeper or larger model (Eg. 4 conv layers and 2 fully connected).  
  
Or simply experiment around with the model parameters. Maybe use different kernel sizes. Try to see if you can further improve the preformance.  
## Step 5: Final testing
  

  
Now that we have a well performing model, we can run the model on the test set and see how it performs on unseen data.  
