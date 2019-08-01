# LeNet-5


卷积神经网络之LeNet-5实现，注意，实际在开发过程中没有用到F6层，主要包括2个卷积层、2个池化层、1个全连接层，外加输入及输出，共7层网络。实际训练时采用最大值池化、双曲正切激活函数，经过8轮迭代训练，手写数字识别准确率即达到99%。

LeNet5 分为7层 Input->C1->S2->C3->S4->C5->F6->Output。参考视觉皮层处理，hubel and wiesel 的猫实验,不同神经细胞对不同图案反应，对应C1->S2->C3->S4->C5用于特征提取。


    Input->C1:
        6个卷积核提取6张特征图
    C2->S2:
        池化既保存主要特征也减少计算量，得到6张特征图
    S2->C3:
        6张图卷积成为16张特征图。第1～6特征图，连接3张图，第7～15特征图，连接4张图，第16特征图，连接6张图。
        首先，一个非完整的连接方式将连接的数量限制在合理的范围内；更重要的是，该连接方式在网络中打破了对称。不同的特征图提取到的特征是不同的，因为它们得到的输入也是不同的。
    C3->S4:
        池化既保存主要特征也减少计算量，得到16张特征图
    S4->C5:
        全连接


![image](https://github.com/bensema/LeNet-5/blob/master/view.png)

![image](https://github.com/bensema/LeNet-5/blob/master/lenet-5.png)



参考资料:
 - [生物学](https://www.coursera.org/lecture/biologyconcept/1-shi-jue-huan-lu-he-dui-shi-jue-xin-hao-de-jia-gong-zheng-he-4dD7m)
 - [lecun](http://yann.lecun.com/)
 - [lecun-98.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
 - [lenet_code](https://github.com/0x7dc/LeNet-5)
