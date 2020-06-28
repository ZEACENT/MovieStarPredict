# -*- coding: utf-8 -*-
import scrapy
import douban.items
import re


class DoubanfinalsSpider(scrapy.Spider):
    name = 'doubanFinals'
    allowed_domains = ['movie.douban.com']
    # 可播放最新热门
    start_urls = [
        'https://movie.douban.com/j/search_subjects?type=movie&tag=%E5%8F%AF%E6%92%AD%E6%94%BE&page_limit=1000',
        'https://movie.douban.com/j/search_subjects?type=movie&tag=%E6%9C%80%E6%96%B0&page_limit=10000',
        'https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&page_limit=10000']
    viewedURL = []
    count = 0
    xpath_time = ['//*[@id="info"]/span[./text()="片长:"]/following::text()[2]',
                  '//*[@id="info"]/span[./text()="片长:"]/following::text()[1]']

    def parse(self, response):
        name = response.xpath('//*[@id="content"]/h1/span[1]/text()').extract()
        if len(name) == 0:  # 简介页
            import json
            rs = json.loads(response.text)
            rs = rs.get('subjects')
            for i in rs:
                if i['url'] not in self.viewedURL:  # 未浏览
                    self.viewedURL.append(i['url'])  # 添加浏览记录
                    yield scrapy.Request(i['url'])
        else:  # 详细页
            try:
                item = douban.items.DoubanItem()
                item['name'] = name[0].strip()
                item['star'] = response.xpath('//*[@id="interest_sectl"]/div[1]/div[2]/strong/text()').extract()[
                    0].strip()
                item['director'] = response.xpath('//*[@id="info"]/span[1]/span[2]/a/text()').extract()[0].strip()
                item['playwright'] = response.xpath('//*[@id="info"]/span[2]/span[2]/a/text()').extract()[0].strip()
                item['lead'] = response.xpath(
                    '//div[@id="info"]/span[@class="actor"]/span[@class="attrs"]/a[position()<=3]/text()').extract()[
                    0].strip()
                item['type'] = response.xpath('//*[@id="info"]/span[@property="v:genre"]/text()').extract()[0]
                item['region'] = \
                    response.xpath('//*[@id="info"]/span[./text()="制片国家/地区:"]/following::text()[1]').extract()[
                        0].strip()
                item['language'] = \
                    response.xpath('//*[@id="info"]/span[./text()="语言:"]/following::text()[1]').extract()[
                        0].strip()
                for x in self.xpath_time:
                    timestr = response.xpath(x).extract()[0].strip()
                    time = re.findall(r"\d*", timestr)[0]
                    if len(time) != 0:
                        item['time'] = time
                        break
                self.count += 1
                print(self.count, item['name'])
                yield item
            except:
                with open('err.txt', 'a') as f:
                    f.write(str(self.count) + str(name) + '\n')
