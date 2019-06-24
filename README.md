# Python-ML-intro [original](https://t.me/dsuse/10755)

《零起点 Python 机器学习快速入门》：复制的简易机器学习入门 Iris 数据集线性回归

作者是一个做中文字库的（字王工作室），自产有[《中华大字库》](http://www.topquant.vip/?cat=16)、TopQuant 足彩分析等，也是一个其他信息行业奠基者的角色，（对我来说）是个先辈，但是他对程序设计的理解有点独特但不是很深刻，书上的图示、算法接口资料还比较齐全，
示例数据集主要讲的是 Iris 爱丽丝花卉子种属数据集，以及一个 CCPP 发电厂电力输出数据集。

全本书讲了 `sklearn` (scikit)、Pandas、Matplotlib 的使用，当然没有提到 Numpy、Numba、OpenCL、OpenMP、PyCUDA 等高性能计算库的使用（“黑箱”教学），但是提了名字

算法没有讲太多算法细节，只是给你几个封装函数做学习、测试数据切分 (`ai_data_cut`)、学习 (`mx_*`)、回归 (`predict` 方法)、测试 (`ai_acc_xed`) （用于判断数据预测的准确率，就是对 test 数据集，准备好预测到的结果，取实际 test 的结果，判断 (误差小于目标 _k_ 的结果项目份数/总份数)）

前面还教你如何进行『分类名称』“矢量”化（就不在这里喷这个名词使用的错误了[[^1]](#fn1)，虽然可能不是他自己最开始用的）...
这里不吐槽任何槽点，但是本书的槽点还是很多的，看上 100 面大概能找到十七八个，不过不得不说对于机器学习入门来说这本书也不错（即使里面有些文字我打算专门吐槽一下...）。
再者，因为我这里绝对没有任何其他的机器学习书了...

###### <small>footnote</small>

<a name="fn1">^1</a> 矢量化：这里是指，对于一个数据表（书上的是 Iris 种属分类）
如果要学习（数据分析）的一项是（可能不可导、不可进行传统统计概率数值运算的）聚合量（[product type](https://en.wikipedia.org/wiki/Product_type)）比如一个 str （char 的 [homogenous product type](http://open-std.org/JTC1/SC22/WG21/docs/papers/2017/p0649r0.pdf)）那就先将其『标号化』、学习，再在回归的时候映射回来
矢量，在物理上是有方向的量[[^2]](#fn2)，数学上（尤其线性代数）矢量化是指[把矩阵转化为线性序列的形式](https://en.wikipedia.org/wiki/Vectorization_(mathematics))，请问这里它的宾语（目标）是指『这个数据表 `pandas.DataFrame`』呢？还是 Python 的 `<built-in type str>` 呢？
不管怎么样，看起来都有点不太对吧？哈？算了....

```matlab
vec(A) = {
  A[1,1], ..., A[m,1],
  A[1,2], ..., A[m,2],
  A[1,n], ..., A[m,n] }
```

简而言之就是把 2x2 矩阵 `{ a b;; c d }` 变成元组 `[ a c b d ]`, 看上面的递推式子也知道了

缺点当然是很明显的，比如 Iris 分类器的例子，作者就这么对三种种属『“矢量化”』了一波（取的『编号』，虽然那实际上是逼近的参数之一.... 都是 1,2,3 这种），Linear regression 的结果很差很辣鸡，只比随机三选一好一点。

然而实际上可以多准确呢？同数据集 kNN 分类器（k-最近邻分类法，最简单的机器学习算法之一，特征聚合映射到 _N_ 维空间判距离上 _k_ 最近邻中数目最多的分类，用作者的话，当然我觉得还行的就是『物以类聚』）学习后给出的判断准确率是 100%！

（但是，实际测试的时候这里准确率都接近 100%... 可能是训练测试数据集不同吧）😑

何况，作者『映射』分类预测结果回来（实际上没有映射回 `str`，但这里和我说的实际上是一个情况）的时候使用的（浮点）算法是直接 truncate(`floor`) 掉小数部分然后 `if else if ...` （如果有可能的话这种风格不如 `switch ...`），还不如四舍五入（使用 Banker's rounding _(to even)_ `round`）好呢（不要忘记了这个『矢量』实际上代表的是一个分类！所以要特殊处理）

作者自己的确是做数据分析的，可是他对编程的理解.... 我觉得真的不是特别值得学习

我可没有随便对自己完全不理解的东西乱说话，至少我能找到一篇文章给我背书：

[Conmajia::CodeProject::前馈全连接神经网络和函数逼近、时间序列预测、手写数字识别](https://www.cnblogs.com/conmajia/p/annt-feed-forward-fully-connected-neural-networks.html#%E9%B8%A2%E5%B0%BE%E8%8A%B1%E5%A4%9A%E7%B1%BB%E5%88%86%E7%B1%BB)这里面没有 Python 书里作者所谓的『矢量化』， 但只是学习模型的不同而已（[线性回归](https://zh.wikipedia.org/wiki/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8) vs. 人工神经网络），不过 Sklearn 有 [one-hot encoding 的实现](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)，作者还是要用『矢量化』定 1,2,3，明明说都到『CPU 加法器』的级别了，却连二进制都没注意到，也真是服气了....


```python
>>> from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit([1, 2, 2, 6])
LabelEncoder()
>>> le.classes_
array([1, 2, 6])
```

<a name="fn2">^2</a> 的时候搞错了... 物理上那个是向量，基本无关的东西

