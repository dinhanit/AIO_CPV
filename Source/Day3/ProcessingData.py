import torchvision
from torchvision import transforms

# Chuẩn bị transform
# Chuẩn bị transform
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.1),
    # transforms.RandomVerticalFlip(p=0.1),
    # transforms.RandomRotation(degrees=15),
    transforms.GaussianBlur(kernel_size = 3),
    # transforms.ColorJitter(
    #     brightness = 0.2, 
    #     contrast = 0.2,
    #     saturation = 0.2,
    #     hue = 0.2
    # ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Tải dữ liệu
train_data = torchvision.datasets.ImageFolder(root='./dataset/train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(root='./dataset/test', transform=test_transforms)
# Số lượng các lớp
num_classes = len(train_data.classes)
# Tên của các lớp
classes_name = train_data.classes
# classes2idx: ánh xạ từ tên lớp sang chỉ số (index)
classes2idx = train_data.class_to_idx
# print(f"Số lượng lớp: {num_classes}")
# print(f"Tên lớp: {classes_name}")
# print(f"Ánh xạ từ tên lớp sang chỉ số: {classes2idx}")
# print("Number of train: ", len(train_data))
# print("Number of test: ", len(test_data))
