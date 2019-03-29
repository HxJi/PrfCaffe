# ResNet 50
## resnet_50_train_val.prototxt
  - Resnet-50 network structure description
  - Batch size: Training: 64, Test: 10
## resnet_34_solver.prototxt
  - Hyperparameters when training the network
  - iter_size: 4 (batch size is 64, 4 * 64 = 256 will be the actual batch size used to update network)
  - test_iter: 10000 (batch size: 10, total test images: 100000, 100000/10 = 10000 iter to test all images)
  - test_interval: 20000 (do the test every 80000 training iteration)
  - display: 10000 (show the information every 20000 iteration)
  - max_iter: 2500000 (imagenet dataset has about 1.2 million images, 1 epoch = 1.2 million/64 = 20000 iter, 2500000 iter = 125 epoch)
  - stepvalue: 600000, 1200000, 1800000 (change the learning rate at 30, 60, 90 epoch)
  - snapshot: 10000 (save the model every half epoch)
  - lr_policy: "multistep" (learning policy is multistep)
  
  Any suggestion on the hyperparameters are welcomed.
  
## resnet50.condor
   - condor job descirption file 
  
## train_resnet50.sh
   - Train from snapshot
