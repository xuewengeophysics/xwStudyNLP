# 爬虫

## 使用Scrapy实现爬虫的主要步骤：

1. 创建一个Scrapy项目。

    scrapy startproject tianya_textclassify
    
2. 定义提取的Item。在items.py文件定义想要提取的实体：

    class TianyaTextclassifyItem(scrapy.Item):
    
        label   = scrapy.Field() # 文档标签
        
        content = scrapy.Field() # 文档内容
        
3. 编写爬取网站的 spider 并提取 Item。创建并修改spiders中蜘蛛文件tianya_ly.py：

    scrapy genspider tianya_ly tianya.cn
   
4. 编写 Item Pipeline 来存储提取到的Item(即数据)。在pipelines.py中定义函数用于存储数据：

    def process_item(self, item, spider):
    
        with open('train_ly.txt', 'a', encoding='utf8') as fw:
        
            fw.write(item['label'] + "," + item['content'] + '\n')
            
        return item
5. 创建执行程序的文件main.py。有两种执行方式：

    (1)：用cmdline执行cmd命令
    
    from scrapy import cmdline
    
    cmdline.execute("scrapy crawl tianya_ly".split())
    
    (2)：常用方式
    
    from scrapy.cmdline import execute
    
    execute(["scrapy", "crawl", "tianya_ly"])
    

## scrapy工作原理图
![image](https://github.com/xuewengeophysics/xwStudyNLP/blob/master/scrapy/images/%E7%88%AC%E8%99%AB.jpg)

## 参考资料
[1] https://scrapy-chs.readthedocs.io/zh_CN/latest/intro/overview.html
