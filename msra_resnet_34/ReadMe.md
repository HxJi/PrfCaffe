# ResNet 34
## resnet_34_train_val.prototxt
  - Resnet-34 network structure description
  - Batch size: Training: 32, Test: 10
## resnet_34_solver.prototxt
  - Hyperparameters when training the network
  - iter_size: 8 (batch size is 32, 8 * 32 = 256 will be the actual batch size used to update network
  - test_iter: 10000 (batch size: 10, total test images: 100000, 100000/10 = 10000 iter to test all images)
  - test_interval: 80000 (do the test every 80000 training iteration)
  - display: 20000 (show the information every 20000 iteration)
  - max_iter: 5120000 (imagenet dataset has about 1.2 million images, 1 epoch = 1.2 million/32 = 40000 iter, 5120000 iter = 128 epoch)
  - stepvalue: 2400000, 3000000, 3600000 (change the learning rate at 60, 75, 90 epoch)
  - snapshot: 20000 (save the model every half epoch)
  - lr_policy: "multistep" (learning policy is multistep)
  
  Any suggestion on the hyperparameters are welcomed.
  
## resnet34.condor
   - condor job descirption file 
  
## train_resnet34.sh
   - Train from snapshot
