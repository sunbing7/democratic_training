# example command to test UAP
################################################################################################################################################
#resnet50
#TARGET_CLASS=174
#test original model
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#test repaired model
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
################################################################################################################################################
#vgg19
#TARGET_CLASS=771
#test original model
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#test repaired model
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
################################################################################################################################################
#googlenet
#TARGET_CLASS=573
#test original model
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#test repaired model
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
################################################################################################################################################
#wideresnet cifar10
TARGET_CLASS=9
#test original model
python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#test repaired model
python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4