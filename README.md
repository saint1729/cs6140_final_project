# Overview
This branch runs training using cifar-10 for the [Zhang et al](https://richzhang.github.io/colorization/) implementation with no fusion, encoder, decoder.

We used [this article](https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/) as a helpful guide

# Requirements
Google colab is pretty awesome, so you just have to go to google colab, upload a
notebook (floyd.ipynb in this directory), and then go Runtime -> Change Runtime
type -> GPU and you're set.

# Getting the data
1. Go to this [kaggle](https://www.kaggle.com/c/cifar-10/data) and grab the data.
2. Upload train.zip to your google drive
3. Upload test.zip to your google drive
4. Uncomment the unzip lines in the 2nd and 3rd code blocks - you will only leave these uncommented for one execution

# Running it
Now you're set to run everything!

Some things to note:
1. The last block is a custom test for Sai's cat image, don't run it unless you have that 32x32 cat in the correct directory
2. The 2nd block mounts the drive and will prompt you to add your auth key
3. Now you're all set to run everything
