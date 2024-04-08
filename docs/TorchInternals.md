# Torch Internals

## <a name="MaxPool"></a> MaxPool layer

### Basics on Max Pooling

Max pooling is a downsampling technique commonly used in convolutional neural networks (CNNs) to reduce the spatial dimensions of an input volume. It is a form of non-linear down-sampling that serves to make the representation smaller and more manageable, and to reduce the number of parameters and computation in the network. Max pooling operates independently on each depth slice of the input and resizes it spatially.

The primary objective of max pooling is to reduce the amount of information in an image while maintaining the essential features necessary for accurate image recognition. This process helps to make the detection of features in input data invariant to scale and orientation changes and also aids in preventing overfitting.

How Max Pooling works?

Max pooling is performed on the convolutional layers of a CNN. It involves sliding a window (often called a filter or kernel) across the input data, similar to the convolution step, but instead of performing a matrix multiplication, max pooling takes the maximum value within the window. This maximum value becomes a single pixel in the new, pooled output. The window is then slid across the input data by a stride of a certain number of pixels, and the process is repeated until the entire input image has been processed.

Typically, the size of the pooling window is 2x2, and the stride with which the window is moved is also 2 pixels. This setup reduces the size of the input by half, both in height and width, effectively reducing the total number of pixels by 75%.

Max Pooling vs other Pooling methods

While max pooling is a popular choice, there are other pooling methods, such as average pooling and L2-norm pooling. Average pooling takes the average of all values in the pooling window, which can result in smoother pooled outputs. L2-norm pooling takes the square root of the sum of squares of the values in the window. Each pooling method has its own advantages and is chosen based on the specific requirements of the model or the type of data being processed.

In practice, max pooling layers are placed after convolutional layers in a CNN. After a convolutional layer extracts features from the input image, the max pooling layer reduces the spatial size of the convolved feature map, keeping only the most salient information. This process is repeated for multiple convolutional and pooling layers, allowing the network to learn a hierarchy of features at various levels of abstraction.

Max pooling is a simple yet effective technique that has been instrumental in the success of CNNs in various applications, particularly in image and video recognition tasks. Its ability to reduce the computational burden while maintaining the essential features has made it a staple component in deep learning architectures.

Challenges with Max Pooling

Despite its benefits, max pooling is not without its challenges. One criticism is that it can sometimes be too aggressive, discarding potentially useful information that could be important for the classification task. Moreover, max pooling is a fixed operation and does not learn from the data, unlike convolutional layers that have learnable parameters.

As a result, some modern CNN architectures have started to move away from traditional max pooling layers, using alternatives like strided convolutions for downsampling or incorporating learnable pooling operations that can adapt to the data.

### Max Pooling layer implementation in torch


```python
class MaxPool2d(_MaxPoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: Tensor):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)
```

