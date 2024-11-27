import scrapy


class HousespiderSpider(scrapy.Spider):
    name = "housespider"
    allowed_domains = ["pararius.com"]
    start_urls = ["https://www.pararius.com/apartments/nederland/"]
    
    # Disable robots.txt for this spider
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
    }

    def parse(self, response):
        houses = response.css('section.listing-search-item')
        for house in houses:
            yield{
                'name': house.css('a.listing-search-item__link--title::text').get().strip()
            }
        
        # Next Page
        next_page = response.css('li.pagination__item.pagination__item--next a::attr(href)').get()
        if next_page is not None:
            next_page_url = 'https://www.pararius.com' + next_page
            yield response.follow(next_page_url, callback = self.parse)
