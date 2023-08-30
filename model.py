from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from efficientnet import EfficientNet

NUM_ACTIONS = 4

class Model:
  def __init__(self, hidden_size):
    self.backbone = EfficientNet(0, has_fc_output=False)
    self.backbone.load_from_pretrained()
    self.fc1 = Linear(1280, hidden_size)
    self.fc2 = Linear(hidden_size, NUM_ACTIONS)
  
  def __call__(self, x:Tensor) -> Tensor:
    x = self.backbone(x)
    x = self.fc1(x).relu()
    x = self.fc2(x)
    return x


if __name__ == '__main__':
  data = Tensor.randn(8, 3, 224, 224)
  model = Model(128)
  out = model(data)
  print("Success!", out.shape)