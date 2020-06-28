使用scrapy对豆瓣3个电影标签的json进行电影爬取：获取json的详细页url，进入url获取必要的信息。

将数据写入mongoDB数据库。

数据分析读取mongoDB数据，将数据进行量化并高斯分布。

训练模型并预测，绘制混淆矩阵，得出recall和precision。

自定义电影信息，让模型预测该电影的得分是否高于6分。结果如下：

![Screen Shot 2020-06-28 at 10.42.27](/GitHub/数据分析实训/Finals/Screen Shot 2020-06-28 at 10.42.27.png)

![Screen Shot 2020-06-28 at 10.42.27](/GitHub/数据分析实训/Finals/Screen Shot 2020-06-28 at 10.42.35.png)