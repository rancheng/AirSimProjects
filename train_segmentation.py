import keras_segmentation

model = keras_segmentation.models.unet.vgg_unet(n_classes=255,  input_height=256, input_width=256)
model.train(
    train_images="Norm_Data/RGB/",
    train_annotations="Norm_Data/Seg/",
    checkpoints_path="Norm_Data/vgg_unet_1",
    epochs=50000
)

