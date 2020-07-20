*Important Notes On Training*

Training models on this AI is very easy compared to Spleeter. However, there a few things that need to be taken into consideration before going forward with training if you want the best results!

1. Training can take very long, I mean from a day to days, to a week straight depending on how many pairs you're using and your PC specs.
2. The better your PC specifications are, the quicker training will be. Also, CPU training is a lot slower than GPU training
3. After much research, training using your GPU is recommended. The more V-RAM, the bigger you can make your batch size.
4. If you choose to train using your GPU, you will need to do research on your GPU first to see if it can be used to train. For example, if you have a high-performance Nvidia Graphics card you'll need to install compatible Cuda drivers for it to work properly. In many cases, this might not actually be worth it due to V-RAM limitations, unless you have a badass GPU with at least 6GB's of V-RAM, or you choose to order a GPU cluster from AWS or another cloud provider (that's an expensive route).
5. The dataset must be comprised of pairs consisting of official instrumentals & official mixes.
6. The pairs MUST be perfectly aligned! Also, the spectrograms must match up evenly, and the pairs should have the same volume, otherwise, training will not be effective and you'll have to start over.
7. I've found in my own experiences that not every official studio instrumental will align with its official mix counterpart. If it doesn't align, that means it's NOT usable! I can't stress this enough. A good example of a pair that wouldn't align for me was "Bring Me The Horizon - Avalanche (Official Instrumental)" & it's official mix with vocals. What I found was the timing was slightly different between the 2 tracks and this rendered it impossible to align. A good example of a track that did align was the instrumental and mix for "Brandon Flowers - Swallow It". For those of you who don't know how to align tracks, know that it's the exact same process people use to extract vocals using an instrumental and mix (vocal extraction isn't necessary for this. Although, knowing if they can be using this classic method, even with a lot of artifacts, makes for a good litmus test for rather or not the pair is a good candidate for training). There are tons of tutorials on YouTube that show how to perfectly align tracks. Also, you can bypass some of that work by making an instrumental and mix out of multitrack stems.
8. I advise against using instrumentals with background vocals and TV tracks as they can undermine the efficiency of the training sessions and model.
9. From my experience, you can use tracks of any quality, as long as they're both the exact same quality (natively). For example, if you have a lossless wav mix, but the instrumental is only a 128kbps mp3, you'll need to convert the lossless mix down to 128kbps so the spectrograms can match up. Don't convert a file up from 128kbps to anything higher like 320kbps, as that won't help at all and will actually make it worse. If you have an instrumental that's 320kbps and a lossless version of it doesn't exist, you're going to have to convert the lossless mix wav file down to 320kbps. With that being said, using high-quality tracks does make training slightly more efficient, but to be honest 320kbps and a lossless wave file won't really make much of a difference at all. The frequencies that are between a 320kbps file and a lossless wav file don't even really contain enough audible vocal data to make an impact. You can use Spek or iZotope to view the spectrograms and to make sure they match up. However, I suggest not using ANY pair below 128kbps. You're better off keeping the dataset between 128-320kbps.
10. When you start training you'll see it go through what are called "Epochs". 100 epochs are the standard, but more can be set if you feel it's necessary. However, if you train on more than 100 epochs you run the risk of either "overfitting" the model (which renders it unable to make accurate predictions), or stagnating the model (basically waste training time). You need to keep a close eye on the "training loss" & "validation loss" numbers. The goal is to get those numbers as low as you can. However, the "training loss" number should always be slightly (and I mean slightly) lower than the "validation loss". If the "validation loss" number is ever significantly higher than the "training loss", that means you're overfitting the model. If the "training loss" & "validation loss" number are both high and stay high after 2 or 3 epochs, then you either need to give it more time to train, or your dataset is too small (you should be training with a bare minimum of 50-75 pairs).
11. A new model spawns after each epoch with the best validation loss, so you can get a good idea as to rather or not it's getting better based on the data you're training it on. However, I recommend you not run conversions during the training process. Load the model to an external device and test it on a separate PC if you must.
12. I highly recommend dedicating a PC to training. Close all applications and clear as much RAM as you can. Even if you use your GPU.
13. Proceed with caution! I don't want anyone burning out their PC's! A.I. training like this is a computationally-intensive process, so make sure your PC is properly cooled and check it's temperature every so often to keep it from overheating. In fact, this is exactly why I use non-gui Linux Ubuntu to do all my training. Using that platform allows me to allocate nearly most of my system resources to training, and training only.
14. Be patient! I feel like I have to stress this because it can take a VERY long time if you're using your CPU, or if your training with a dataset consisting of more than 200 pairs on a GPU. This is why you want to prepare a good dataset before starting. The model my friend and I created took nearly a week and a half straight to complete! This means not running any resource sucking applications in the background, like browsers and music players. Prepare to essentially surrender usage of that PC until the training has completed.


Note: First and foremost, please read the *Important Notes On Training* above before starting any of this!

1. Download the Vocal Remover package from here that includes the training script - https://github.com/tsurumeso/vocal-remover
Training Instructions

2. Place your pairs accordingly in the following directories in the main vocal-remover folder

dataset/
+- instrumentals/
| +- 01_foo_inst.wav
| +- 02_bar_inst.mp3
| +- ...
+- mixtures/
| +- 01_foo_mix.wav
| +- 02_bar_mix.mp3
| +- ...

3. Run one of the following commands:

-To run the training process on your CPU, run the following -

~To finetune the baseline model, run the following -

python train.py -i dataset/instrumentals -m dataset/mixtures -P models/baseline.pth -M 0.5 -g 0

~To create a model from scratch, run the following -

python train.py -i dataset/instrumentals -m dataset/mixtures -M 0.5

NOTE: This is not guaranteed to work on on graphics cards 4GB's and below! You might need to change your batch settings.

4. Once you see the following on the command prompt it means the training process has begun:

# epoch 0
* inner epoch 0