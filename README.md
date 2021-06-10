# AMATH563_Project
The following gives a brief overview of the code and parameters that can bet set.

## AMATH563_Training_Project.ipynb
This jupyter notebook trains and validates the image classification models. The following hyperparameters are available to tune
 * dataset_id: 'Grozi', 'Freiburg'
 * model_select: 'AlexNet', 'vgg16', 'resnet', 'mobilenet'
 * optim_select: 'adam', 'sgd'
 * batch_size
 * pre_train: whether pre-trained weights from ImageNet should be used
 * fine_tune: if True, only retrain final two layers. if False, train all fc layers
 * init: if True, retrain all layers (overrides fine_tune)

## AMATH563_Detection_Model.ipynb
This jupyter notebook trains and validates the Faster R-CNN object detection model. Note that the object detection can only be performed on the Grozi testing dataset. The following hyperparameters are available to tune
 * weight_update: if True, use our best performing model weights for the backbone
 * req_grad: if True, allow our backbone weights to be trained (only has impact if weight_update is True)
 * pre_train: if True, load base Faster R-CNN model trained on the COCO dataset
 * train_backbone_layers: int between 0 and 5 for how many backbone layers are able to be trained
 * scheduler: 'plateau' (ReduceLROnPlateau) or 'step' (StepLR)

## AMATH563_Detection_Evaluation.ipynb
This jupyter notebook runs a detection model against the validation set from the Grozi testing set and outputs annotated images. Can take in any detection model
