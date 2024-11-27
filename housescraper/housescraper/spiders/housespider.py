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
            relative_url = house.css('a.listing-search-item__link--title::attr(href)').get()
            house_url = 'https://www.pararius.com' + relative_url
            yield scrapy.Request(house_url, callback=self.parse_house_page)
        
        # Next Page
        next_page = response.css('li.pagination__item.pagination__item--next a::attr(href)').get()
        if next_page is not None:
            next_page_url = 'https://www.pararius.com' + next_page
            yield response.follow(next_page_url, callback = self.parse)
    
    def parse_house_page(self, response):
        house = response.css('div.listing-detail-summary')
        transfer = response.css('section.page__details--transfer') 

        yield{
            'Name': house.css('h1.listing-detail-summary__title::text').get(),
            'Address': house.css('div.listing-detail-summary__location::text').get(),
            'Price': house.css('span.listing-detail-summary__price-main::text').get().strip(),
            # 'Deposite': transfer.css('dd.listing-features__description--deposit span.listing-features__main-description::text').get(),
            # 'Service Cost': transfer.css('ul.listing-features__sub-description li::text').get(),
            'Area': house.css('li.illustrated-features__item--surface-area::text').get().strip(),
            'Number of Rooms': house.css('li.illustrated-features__item--number-of-rooms::text').get().strip(),
            'Interior': house.css('li.illustrated-features__item--interior::text').get().strip(),
            'Offered Since': transfer.css('dd.listing-features__description--offered_since span.listing-features__main-description::text').get(),
            'Status': transfer.css('dd.listing-features__description--status span.listing-features__main-description::text').get(),
            'Available': transfer.css('dd.listing-features__description--acceptance span.listing-features__main-description::text').get(),
            'Rental Agreement': transfer.css('dd.listing-features__description--contract_duration span.listing-features__main-description::text').get(),
            'Duration':  transfer.css('dd.listing-features__description--contract_duration_min_max span.listing-features__main-description::text').get(),




        }

