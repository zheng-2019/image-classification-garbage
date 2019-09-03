class DefaultConfigs(object):
    #1.string parameters
    train_data = "/home/ltdev/lingtian/jhzheng/s3data/image-classification-garbage/data/train_color.txt"
    test_data = "/home/ltdev/lingtian/jhzheng/s3data/image-classification-garbage/data/test_color.txt"
    val_data = "/home/ltdev/lingtian/jhzheng/s3data/image-classification-garbage/data/val_color.txt"
    #model name :densenet169,densenet161,densenet121,densenet201   resnet101,resnet152,resnet50
    model_name = "densenet121"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"

    num_garbages = 40  #色彩
    #2.numeric parameters
    epochs = 200
    batch_size = 4
    img_height = 300
    img_weight = 300
    seed = 888
    lr = 5e-2
    lr_decay = 1e-4
    weight_decay = 1e-4
    randomsimple = True#权重采样   
    resume = True
    testwithgpu = True

config = DefaultConfigs()
