import dataset
import torchvision.models as models
import torch

ds = dataset.create_wall_dataloader("/scratch/DL24FA/train",probing=False,batch_size=1)
resnet50 = models.resnet50(weights='DEFAULT').cuda()
# list(resnet50.children()))
for data in ds:
	print(data)
	print(data.states[:,0], data.actions[:,0])
	print(data.states.shape,data.actions.shape)
	print(resnet50(torch.concat([data.states[:,0], torch.zeros(65,65)], dim=0).cuda()))
	break
