# Message Content

## Part 1

#Python 四舍五入能『优化“矢量化”后的预测结果准确率』一看可能不是正确答案，但是我会用实践测试一下它是不是正确的

首先看看这个 [wiki 页面](https://zh.wikipedia.org/wiki/%E5%AE%89%E5%BE%B7%E6%A3%AE%E9%B8%A2%E5%B0%BE%E8%8A%B1%E5%8D%89%E6%95%B0%E6%8D%AE%E9%9B%86)，我们提取 HTML 表格数据（下面的编程风格不值得学习，复用性也很差，是我自己瞎堆的，因为我不熟悉 JQuery,,,）

```es6
let table = $('table.wikitable.sortable')
  .filter((_, e) => e.firstElementChild.innerText === "费雪鸢尾花卉数据集").get(0);

let data_body = $(table).find('tbody').get(0);

//

{ let result = [];
  for (let t of data_body.children) {
    let [x, y, z, w, o] = t.children;
    result.push(...[x,y,z,w,o]);
  }}
```

🙈 不好意思，这个是 ~~矢量化~~

```es6
let result = $(data_body.children)
  .map((_, t) => [($(t.children).slice(0,2+1)
    .map((_, c) => Number.parseFloat(c.innerText)))
      .get().concat(t.children[4].innerText)]);
```

刚才本来用的是 array expansion（其实有点 deficient）`[...xs, a]` 的，后来发现 JQuery 的 map 好像会自动 flatten... 就换成这样了
`slice` 的 `2+1` 是因为 JQuery 的 slice 切片是右闭（exclusive）区间，`2` 是输出的最后一个 index

然后我们输出 CSV，上 Python Pandas + Sklearn 分析吧，首先我们已经确认，数据里没有特殊字符（和 CSV 语义冲突的）需要处理

```es6
result.map((_, t) => t.join(',')).get().join("\n")
```

好的程序应该反映出其所处理的数据的结构，这里 `map` 里的闭包输出每个 row 的 `toCsvString` 操作结果（join each column）；最后每行再合并（`join`）就生成了 CSV 嵌套结构

复制下对象、注意去掉两边的 `""` dquote，然后保存到本地（也可以 `console.log` 结果字符串）

## File _Iris.csv_

#ML #Python Iris 数据集

```csv
x,y,z,w
5.1,3.5,1.4,setosa
4.9,3,1.4,setosa
4.7,3.2,1.3,setosa
4.6,3.1,1.5,setosa
5,3.6,1.4,setosa
5.4,3.9,1.7,setosa
4.6,3.4,1.4,setosa
5,3.4,1.5,setosa
4.4,2.9,1.4,setosa
4.9,3.1,1.5,setosa
5.4,3.7,1.5,setosa
4.8,3.4,1.6,setosa
4.8,3,1.4,setosa
4.3,3,1.1,setosa
5.8,4,1.2,setosa
5.7,4.4,1.5,setosa
5.4,3.9,1.3,setosa
5.1,3.5,1.4,setosa
5.7,3.8,1.7,setosa
5.1,3.8,1.5,setosa
5.4,3.4,1.7,setosa
5.1,3.7,1.5,setosa
4.6,3.6,1,setosa
5.1,3.3,1.7,setosa
4.8,3.4,1.9,setosa
5,3,1.6,setosa
5,3.4,1.6,setosa
5.2,3.5,1.5,setosa
5.2,3.4,1.4,setosa
4.7,3.2,1.6,setosa
4.8,3.1,1.6,setosa
5.4,3.4,1.5,setosa
5.2,4.1,1.5,setosa
5.5,4.2,1.4,setosa
4.9,3.1,1.5,setosa
5,3.2,1.2,setosa
5.5,3.5,1.3,setosa
4.9,3.6,1.4,setosa
4.4,3,1.3,setosa
5.1,3.4,1.5,setosa
5,3.5,1.3,setosa
4.5,2.3,1.3,setosa
4.4,3.2,1.3,setosa
5,3.5,1.6,setosa
5.1,3.8,1.9,setosa
4.8,3,1.4,setosa
5.1,3.8,1.6,setosa
4.6,3.2,1.4,setosa
5.3,3.7,1.5,setosa
5,3.3,1.4,setosa
7,3.2,4.7,versicolor
6.4,3.2,4.5,versicolor
6.9,3.1,4.9,versicolor
5.5,2.3,4,versicolor
6.5,2.8,4.6,versicolor
5.7,2.8,4.5,versicolor
6.3,3.3,4.7,versicolor
4.9,2.4,3.3,versicolor
6.6,2.9,4.6,versicolor
5.2,2.7,3.9,versicolor
5,2,3.5,versicolor
5.9,3,4.2,versicolor
6,2.2,4,versicolor
6.1,2.9,4.7,versicolor
5.6,2.9,3.6,versicolor
6.7,3.1,4.4,versicolor
5.6,3,4.5,versicolor
5.8,2.7,4.1,versicolor
6.2,2.2,4.5,versicolor
5.6,2.5,3.9,versicolor
5.9,3.2,4.8,versicolor
6.1,2.8,4,versicolor
6.3,2.5,4.9,versicolor
6.1,2.8,4.7,versicolor
6.4,2.9,4.3,versicolor
6.6,3,4.4,versicolor
6.8,2.8,4.8,versicolor
6.7,3,5,versicolor
6,2.9,4.5,versicolor
5.7,2.6,3.5,versicolor
5.5,2.4,3.8,versicolor
5.5,2.4,3.7,versicolor
5.8,2.7,3.9,versicolor
6,2.7,5.1,versicolor
5.4,3,4.5,versicolor
6,3.4,4.5,versicolor
6.7,3.1,4.7,versicolor
6.3,2.3,4.4,versicolor
5.6,3,4.1,versicolor
5.5,2.5,4,versicolor
5.5,2.6,4.4,versicolor
6.1,3,4.6,versicolor
5.8,2.6,4,versicolor
5,2.3,3.3,versicolor
5.6,2.7,4.2,versicolor
5.7,3,4.2,versicolor
5.7,2.9,4.2,versicolor
6.2,2.9,4.3,versicolor
5.1,2.5,3,versicolor
5.7,2.8,4.1,versicolor
6.3,3.3,6,virginica
5.8,2.7,5.1,virginica
7.1,3,5.9,virginica
6.3,2.9,5.6,virginica
6.5,3,5.8,virginica
7.6,3,6.6,virginica
4.9,2.5,4.5,virginica
7.3,2.9,6.3,virginica
6.7,2.5,5.8,virginica
7.2,3.6,6.1,virginica
6.5,3.2,5.1,virginica
6.4,2.7,5.3,virginica
6.8,3,5.5,virginica
5.7,2.5,5,virginica
5.8,2.8,5.1,virginica
6.4,3.2,5.3,virginica
6.5,3,5.5,virginica
7.7,3.8,6.7,virginica
7.7,2.6,6.9,virginica
6,2.2,5,virginica
6.9,3.2,5.7,virginica
5.6,2.8,4.9,virginica
7.7,2.8,6.7,virginica
6.3,2.7,4.9,virginica
6.7,3.3,5.7,virginica
7.2,3.2,6,virginica
6.2,2.8,4.8,virginica
6.1,3,4.9,virginica
6.4,2.8,5.6,virginica
7.2,3,5.8,virginica
7.4,2.8,6.1,virginica
7.9,3.8,6.4,virginica
6.4,2.8,5.6,virginica
6.3,2.8,5.1,virginica
6.1,2.6,5.6,virginica
7.7,3,6.1,virginica
6.3,3.4,5.6,virginica
6.4,3.1,5.5,virginica
6,3,4.8,virginica
6.9,3.1,5.4,virginica
6.7,3.1,5.6,virginica
6.9,3.1,5.1,virginica
5.8,2.7,5.1,virginica
6.8,3.2,5.9,virginica
6.7,3.3,5.7,virginica
6.7,3,5.2,virginica
6.3,2.5,5,virginica
6.5,3,5.2,virginica
6.2,3.4,5.4,virginica
5.9,3,5.1,virginica
```

## Part 2

#ML #Python 接下来安装一点用于数据分析的东西

```bash
pip3 install --user --compile -U pandas matplotlib sklearn
```

如果有权限你也可以去掉 `--user` flag，`-U` 的意思是如果有的话安装更新
我们使用 Python 3，当然，要是有 IPython 和 Spyder 这种为数据分析 REPL 优化的东西就更好了，不过下面直接用 IPython3

现代机器学习算法主要包含概念学习、规则学习、基于解释学习、基于实例学习、遗传学习、强化学习、贝叶丝学习、神经网络、决策树等分类，当然我也不是很清楚都有啥大区别，反正就是数据分析就对了。

然后函数式编程一般有以下特征 #FP
+ 闭包、高阶函数、函数作为值
+ 基于 Lambda calculus、柯里化(currying)
+ 纯函数无副作用、引用透明、惰性求值
+ 模式匹配，递归作为程序控制基本方式
.... 跑题了，打住。

机器学习某度词条里说是

>”机器学习(Machine Learning, ML)是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。“

其实你记住，无非就是数学的统计、概率、函数、微积分，然后加上计算机科学的算法、软件工程什么的最多了，其他的我们这些渣渣也用不到。

然后我现在”尤其“喜欢神经网络，不过很可惜还没有入门😑 不是因为神经网络复杂（一般我们手动会讨论的基本都是些小网络和网络结构什么的，然后还会讨论学习算法，神经元基本模型本身并不复杂）

在等待下载的时候可以先定义几个函数，首先我们要用 `sklearn` 里实现的 [linear regression](https://zh.wikipedia.org/wiki/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8) 学习实现
我可不擅长黑箱教学是̶~~因̶为̶话̶太̶多̶的̶缘̶故̶么̶~~

说起来这个教程居然还提到了 Python 有 Complex numbers（数学上的复数，包含实数和虚数单位）算是『惊喜』吗？

然后我们有了算法输入了（按照某学姐的说法，神经网络就是一个带有未知参数的程序，那机器学习的很多算法也差不多，这里是有仨输入仨未知系数一输出的数学函数，我们的学习算法会”猜“一个最优化的系数组）
那先按某个比率 _k_ 切分学习数据和测试数据来的，定义函数 `splitTrainingTestData`

```python
from pandas import DataFrame
```

```python
def splitTrainingTestData(ts: DataFrame, kratio: int = 60) -> (DataFrame, DataFrame):
  assert kraito <= 100, "kratio must be valid integral percentage"  
  (txs, drs) = ts.values.tolist(), round(100 / kratio)
  training = txs[::drs]; testset = [x for x in txs if x not in training]
  return training, testset
```

... 刚才把 `kraito` 放到 `ts` 前面去了，Haskell 写多了的结果... （柯里化）
有点秀而且华而不实甚至有问题（跑）其实实践中不应该这么做，但是这里不是实践，所以推荐大家打开思路去实现算法
然后考虑一下最后的验证，测试预测准确率的函数 `verifyRegressionAccuracy`
当然，为了优雅性这时候可以给表格加上头（header）了，就是在它的第一行写上 `x,y,z,w`... 就是给数据列命名

```python
def verifyRegressionAccuracy(ts: DataFrame, emax: float = 0.1, npredict = 'predict', ntruth = 'real') -> float:
  predicteds, truths = ts[npredict], ts[ntruth]
  acceptables = [t for (i, t) in enumerate(predicteds) if abs(t - truths[i]) <emax]
  return len(acceptables) / len(predicteds)
```

总之就是先弄出 acceptable results 然后 `len` 取分数啦
然后简单 play 一下，Python 表达起来也很方便

+ （书上的例子）Pip 已安装软件包

```python
from pandas import DataFrame
from pkg_resources import working_set
import re

dists = [(d.project_name, d.version) for d in working_set]

pkgs = DataFrame()
pkgs['name'] = [n for (n, _) in dists]
pkgs['version'] = [v for (_, v) in dists]

pkgs.tail(5)
pkgs.sort_values('version').head(10)
pkgs.describe()

pkgs['version'].describe()
pkgs['version'].map(lambda v: int(re.sub('rc|dev|\.','', v))).astype(int)
```

隐隐约约会感受到 Python 的确是数据处理很方便，有不少便利语法（slice 和复制的 `[:]` 下标、list comprehension `[... for ... in ... if ...]`）

但是！在其他新兴的编程语言（比如 Kotlin）里！这些功能基本都是可以直接用面向对象多态重载、运算符重载、高阶函数定义出来的！ 🤔

+ 简单的数据折线图(plot)

```python
from matplotlib import pyplot as plot
from pandas import Series
from math import sin
```

然后一个简单的 `sin` series, 很经典吧？

```python
xs = range(0,200, 3)
```

很可惜没有使用 `float` 的 range... 也罢

```python
ys = [sin(x) for x in xs]
```

但是数据不容易看见，谁对着那堆数字有感觉啊！
数据可视化！

`plot.style.available` ... 看看可以有啥绘图风格

```python
plot.style.use('Solarize_Light2')
```

然后

```python
Series(ys).plot()
```
或者
```python
plot.plot(ys)
```

均可，之后可以选择 `plot.show()` （貌似没有用，因为是 IPython CLI 而不是 Spyder 下）
和 `plot.savefig('path.png')`


—

## Part 3

然后可以开始了

```python
from pandas import Series, read_csv

iris = read_csv('Iris.csv', encoding='utf-8', parse_dates=[], index_col=False)
```

然后我们就有了 Iris 数据集的工作实例(working set)（跑

先看看

In \[39]: `iris.describe()`

Out\[39]: 
```r
                x           y           z
count  150.000000  150.000000  150.000000
mean     5.843333    3.057333    3.758000
std      0.828066    0.435866    1.765298
min      4.300000    2.000000    1.000000
25%      5.100000    2.800000    1.600000
50%      5.800000    3.000000    4.350000
75%      6.400000    3.300000    5.100000
max      7.900000    4.400000    6.900000
```

+ _count_ 是整个列表的求和
+ _mean_ 是平均值、_std_ 是方差
+ _min_，_max_ 肯定都知道
+ 50% 是中位数，其他 ?% 依此类推

In \[40]: `iris\['w'].value_counts()`

Out\[40]: 
```matlab
versicolor    50
virginica     50
setosa        50
Name: w, dtype: int64
```

In \[43]: `iris.tail(5)`

Out\[43]: 
```matlab
       x    y    z          w
145  6.7  3.0  5.2  virginica
146  6.3  2.5  5.0  virginica
147  6.5  3.0  5.2  virginica
148  6.2  3.4  5.4  virginica
149  5.9  3.0  5.1  virginica
```

好了，已经说明问题了，现在我们要根据 _f_(_x_, _y_, _z_) 和它的结果 _w_ 学习 _f_ 这个曲线

不过有一个问题，就是 w 不是数值怎么量化，那我们就先看看按 0, 1, 2, ... 『文本 _矢_ 量化』分会有怎么样的结果（怎么感觉和以前我把 Unification 当成泛化(Generalization) 的时候一样）

```python
iris['id'] = Series().astype(int)

def vectorize(w,i, cname='w', cid='id', iris=iris): iris.loc[iris[cname]== w, cid] = i
```

In \[3]: `vectorize('setosa', 0)`

In \[4]: `iris.head(5)`

Out\[4]: 
```matlab
     x    y    z       w   id
0  5.1  3.5  1.4  setosa  0.0
1  4.9  3.0  1.4  setosa  0.0
2  4.7  3.2  1.3  setosa  0.0
3  4.6  3.1  1.5  setosa  0.0
4  5.0  3.6  1.4  setosa  0.0
```

```python
vectorize('versicolor', 1)
vectorize('virginica', 2)
```

OK, 这就是”矢“量化
```python
iris.to_csv('Iris_vectorized.csv', index=False)

print (open('Iris_vectorized.csv').read())
```

In \[17]: `iris['id'].value_counts()`

Out\[17]: 
```matlab
2.0    50
1.0    50
0.0    50
Name: id, dtype: int64
```

然后进行数据预处理切分，之前的算法因为还有点偏差所以就不用了

```python
from sklearn.model_selection import train_test_split
````

切分数据

```python
iris_ds = iris.copy()

trainset, testset, trainsetid, testsetid = train_test_split(iris_ds, iris_ds['id'], train_size = 0.6)

del trainset['w']

trainset.describe()
trainsetid.describe()
```

就不填 `random_state` 了
然后直接用 `sklearn` 的算法学习

```python
from sklearn.linear_model import LinearRegression
from math import floor

lreg = LinearRegression()
lreg.fit(trainset, trainsetid)
```

我们刚才”学习“了这些数据，看看我们能得到什么：

In \[50]: `testset.head(3)`

Out\[50]: 
```matlab
      x    y    z           w   id
91  6.1  3.0  4.6  versicolor  1.0
73  6.1  2.8  4.7  versicolor  1.0
79  5.7  2.6  3.5  versicolor  1.0
```

预测一下（”下面“都是 `numpy` 高性能计算的，这还有一层封装... 不过 `pandas` 也够了）

当然，只是给它一个二维矩阵也可以的 `[[x,y,z]]`，不过好像要有名字...

```python
testset_truth = testset['w']
del testset['w']
```
—
```python
testset['predict'] = lreg.predict(testset)
```

然后手工看看结果

```python
value_map = {-1: 'setosa', 0: 'setosa', 1: 'versicolor', 2: 'virginica'}

testset['guess'] = testset['predict'].map(lambda x: value_map[floor(x)])
testset['w'] = testset_truth
```

🤔
