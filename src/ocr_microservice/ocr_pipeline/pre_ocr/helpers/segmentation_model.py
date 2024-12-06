from segformer_pytorch import Segformer

segmentation_model = Segformer(
    dims=(32, 64, 160, 256),
    heads=(1, 2, 5, 8),
    ff_expansion=(8, 8, 4, 4),
    reduction_ratio=(8, 4, 2, 1),
    num_layers=2,
    decoder_dim=256,
    num_classes=1  
    )