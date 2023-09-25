# PyTorch
Learning to use Torch

# Images using PyTorch
Images - Collection of scalars arranged in a regular grid with height and width. 
A single scalar per grid point denotes a Grayscale image
3 scalars per grid point denotes an RGB image

# Building Neural Networks in PyTorch
1. Create a NeuralNetwork class that extends the base class - <code>nn.Module</code>
2. Define Layers as attributes
3. Implement the <code>forward()</code> method for the forward pass

### A Dummy Neural Network
<code>
class NeuralNetwork(nn.Module):<br>
    def __init__(self):<br>
        super(self,NeuralNetwork).__init__()<br>
        self.layer = None<br>

    def forward(self, t):
        t = self.layer(t)
        return t 
</code>