# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
# scrapy startproject douban
# scrapy genspider [options] <name> <domain>
import scrapy


class DoubanItem(scrapy.Item):
    name = scrapy.Field()  # 电影名称
    star = scrapy.Field()  # 评分
    director = scrapy.Field()  # 导演
    playwright = scrapy.Field()  # 编剧
    lead = scrapy.Field()  # 主演
    type = scrapy.Field()  # 类型
    region = scrapy.Field()  # 地区
    language = scrapy.Field()  # 语言
    time = scrapy.Field()  # 片长
