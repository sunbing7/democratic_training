
### notes on parameter selection ###
# split_layers: you may need to modify split_model() method and set your split_layers on new models introduced to this project
# num_iterations: number of epoches to finetune
# num_batches: set this number based on number of clean samples to use during finetuning process
# alpha: importance of low entropy sample over clean sample during repair_ae process
# ae_iter: number of iterations to generate low entropy samples on the fly. Larger number will increase execution time. Adjust this number if the repair performance is not ideal. WIP on an algorithm to find the most suitable learning_rate automatically.
# option: 1) repair_ae: finetune with low entropy samples; 2) repair: finetune with clean samples to boost the accuracy after repair_ae base on needs.
# learning_rate: you may need to adjust learning rate for your own model. WIP on an algorithm to find the most suitable learning_rate automatically.
################################################################################################################################################
#resnet50
#TARGET_CLASS=174
#python analyze_input.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_iter=5 --target_class=$TARGET_CLASS
#python analyze_input.py --option=repair --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_ae_repaired.pth --learning_rate=0.0001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=$TARGET_CLASS

################################################################################################################################################
#vgg19
#TARGET_CLASS=771
#python analyze_input.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --learning_rate=0.001 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_iter=15 --target_class=$TARGET_CLASS
#python analyze_input.py --option=repair --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet_ae_repaired.pth --learning_rate=0.0001 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=800 --ngpu=1 --workers=4 --target_class=$TARGET_CLASS

################################################################################################################################################
#googlenet
#TARGET_CLASS=573
#python analyze_input.py --option=repair_ae --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --learning_rate=0.001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_iter=10 --target_class=$TARGET_CLASS
#python analyze_input.py --option=repair --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired.pth --learning_rate=0.0001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=$TARGET_CLASS

################################################################################################################################################
#wideresnet cifar10
TARGET_CLASS=9
python analyze_input.py --option=repair_ae --dataset=cifar10 --arch=wideresnet --model_name=wideresnet_cifar10.pth --learning_rate=0.001 --split_layers 6 --seed=123 --num_iterations=50 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=64 --ngpu=1 --workers=4 --alpha=0.9 --ae_iter=5 --target_class=$TARGET_CLASS
python analyze_input.py --option=repair --dataset=cifar10 --arch=wideresnet --model_name=wideresnet_cifar10_ae_repaired.pth --learning_rate=0.00001 --split_layers 6 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=64 --ngpu=1 --workers=4 --target_class=$TARGET_CLASS