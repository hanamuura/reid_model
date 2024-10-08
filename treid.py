import torchreid
import torch

def build_and_train_model(pretrained_weights=None):
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )
    model = model.cuda()

    if pretrained_weights:
        checkpoint = torch.load(pretrained_weights)
        model.load_state_dict(checkpoint["state_dict"])

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir="log/resnet50",
        max_epoch=10,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )

    return model


def load_trained_model(model_path):
    checkpoint = torch.load(model_path)

    state_dict = checkpoint["state_dict"]

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=751,
        loss="softmax",
        pretrained=False
    )
    model = model.cuda()

    model.load_state_dict(state_dict)

    return model

if __name__ == "__main__":
    pretrained_weights = "log/resnet50/model/model.pth.tar-10"
    model = build_and_train_model(pretrained_weights)
