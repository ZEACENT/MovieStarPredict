# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import json
from scrapy.utils.project import get_project_settings
import pymongo


class DoubanPipeline:
    def __init__(self):
        self.file = open('movies.json', 'wb')
        settings = get_project_settings()
        host = settings['MONGODB_HOST']
        port = settings['MONGODB_PORT']
        dbname = settings['MONGODB_DBNAME']
        sheetname = settings['MONGODB_SHEETNAME']
        client = pymongo.MongoClient(host, port)
        mydb = client[dbname]
        self.sheet = mydb[sheetname]

    def open_spider(self, spider):
        self.file.write(b'[\n')  # 写json头
        self.sheet.delete_many({})  # 清空数据

    def process_item(self, item, spider):
        data = dict(item)
        json_str = json.dumps(data) + ',\n'
        self.file.write(json_str.encode('utf-8'))
        self.sheet.insert(data)
        return item

    def close_spider(self, spider):
        self.file.seek(-2, 2)
        self.file.write(b'\n]')
        self.file.close()
