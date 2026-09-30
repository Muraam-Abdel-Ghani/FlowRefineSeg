# FlowRefineSeg
Source code for running FlowRefineSeg, our holistic surgical segmentation model built on a PyConvResNet backbone with gaussian and temporal refinement, published in MICAD2026.

FlowRefineSeg model is divided into the base model, defined in the PyConvFASLLinearGauss.py file, and the Refinery module, defined in RefineryModuleWithFlow.py. For the base model, the backbone class is defined in pyconvresnet.py. For the Refinery module, SpyNet is defined in SpyNet.py. It is recommended that all code files be placed under the same directory when running the code, otherwise feel free to change the paths/import definitions in each file as needed to locate the class files. 

In training, the base model is trained first separately (train.py), then the best checkpoint is used and loaded to train the Refinery module (train-SPYNetOpticFlow.py).


Trained Model Checkpoints:
- EndoVis18 Holistic:   [MuraamAbdelGhani/FlowRefineSeg-Endo18Holistic](https://huggingface.co/MuraamAbdelGhani/FlowRefineSeg-Endo18Holistic/tree/main)
- CholecSeg8k: [MuraamAbdelGhani/FlowRefineSeg-CholecSeg8k](https://huggingface.co/MuraamAbdelGhani/FlowRefineSeg-CholecSeg8k/tree/main)
