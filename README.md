# LeNet-5


卷积神经网络之LeNet-5实现，代码参考了 LeNet-5（https://github.com/0x7dc/LeNet-5) ,tiny-cnn（https://github.com/nyanp/tiny-cnn) 。

注意，实际在开发过程中没有用到F6层，主要包括2个卷积层、2个池化层、1个全连接层，外加输入及输出，共7层网络。实际训练时采用最大值池化、双曲正切激活函数，经过8轮迭代训练，手写数字识别准确率即达到99%。

LeNet5 分为7层 Input->C1->S2->C3->S4->C5->F6->Output。参考视觉皮层处理，hubel and wiesel 的猫实验,不同神经细胞对不同图案反应，对应C1->S2->C3->S4->C5用于特征提取。


![image](https://github.com/bensema/LeNet-5/view.png)

![image](https://github.com/bensema/LeNet-5/lenet-5.png)



参考资料:
 - [生物学](https://www.coursera.org/lecture/biologyconcept/1-shi-jue-huan-lu-he-dui-shi-jue-xin-hao-de-jia-gong-zheng-he-4dD7m)
 - [lecun](http://yann.lecun.com/)
 - [lecun-98.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
