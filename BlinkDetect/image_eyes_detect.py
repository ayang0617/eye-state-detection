import torch
import cv2
from torchvision import transforms
from model import Lenet

def main():
    image = cv2.imread("/Users/ayang/PycharmProjects/pythonProject/EyesImageDatabase/closed_eyes/closed_eyes012774.jpg", 1)
    image = cv2.resize(image, (50, 50))
    image_to_tensor = transforms.Compose([transforms.ToTensor()])
    img_tensor = image_to_tensor(image)
    img_tensor = img_tensor.view(1, 3, 50, 50)

    model = Lenet()
    pthfile = "/Users/ayang/PycharmProjects/pythonProject/bestmodel.pth"
    model.load_state_dict(torch.load(pthfile))
    model.eval()
    output = model(img_tensor)
    prediction = torch.max(output, 1)
    prediction = prediction[1].numpy()[0]
    info = {0: "closed_eyes", 1: "opened_eyes"}
    prediction = info[prediction]
    result = 'prediction is : ' + prediction
    print(result)



if __name__ == '__main__':
    main()