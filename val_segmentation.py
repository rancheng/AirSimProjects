import keras_segmentation

chkpt_path = "./Norm_Data/vgg_unet_1"
# model = keras_segmentation.models.unet.vgg_unet(n_classes=255,  input_height=256, input_width=256)
model = keras_segmentation.predict.model_from_checkpoint_path(checkpoints_path=chkpt_path)
out = model.predict_segmentation(
    inp="./Norm_Data/Experiments/1561381891.351139.png",
    out_fname="./1561381891.351139.png"
)