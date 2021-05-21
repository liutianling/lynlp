import torch
import torch.nn as nn
import torchvision


def get_race_model(device, path):
    model_fair_7 = torchvision.models.resnet34(pretrained=False)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    # model_fair_7.load_state_dict(torch.load('fair_face_models/fairface_alldata_20191111.pt'))
    model_fair_7.load_state_dict(torch.load(path))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()
    return model_fair_7


race_device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
race_model = get_race_model(race_device, './faceRec/deploy/models/FairFace/res34_fair_align_multi_7_20190809.pt')
print("Finish load age model")

dummy_input = torch.randn(1, 3, 224, 224, device=race_device)
torch.onnx.export(race_model, dummy_input, "fairface.onnx")
