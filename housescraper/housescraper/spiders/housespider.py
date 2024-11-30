import scrapy
from housescraper.items import HousescraperItem

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
        house_item = HousescraperItem()
        house = response.css('div.listing-detail-summary')
        transfer = response.css('section.page__details--transfer') 
        construction = response.css('section.page__details--construction')
        layout = response.css('section.page__details--layout')
        outdoor = response.css('section.page__details--outdoor')
        energy = response.css('section.page__details--energy')
        condition = response.css('section.page__details--contract_conditions')
        agent = response.css('section.agent-summary')
        agent_page = 'https://www.pararius.com' + 'a.agent-summary__title-link::attr(href)'

        house_item['House Name']= house.css('h1.listing-detail-summary__title::text').get()
        house_item['Address']= house.css('div.listing-detail-summary__location::text').get()
        house_item['Price']= house.css('span.listing-detail-summary__price-main::text').get() # Remove Euros sign and comma
        house_item['Deposite']= transfer.css('dd.listing-features__description--deposit span.listing-features__main-description::text').get() # Remove Euros sign and comma
        house_item['Service Cost']= transfer.css('ul.listing-features__sub-description li::text').get()
        house_item['Area']= house.css('li.illustrated-features__item--surface-area::text').get()
        house_item['Number of Rooms']= house.css('li.illustrated-features__item--number-of-rooms::text').get() # Strip if not None
        house_item['Interior']= house.css('li.illustrated-features__item--interior::text').get() # strip if not None
        house_item['Dewlling Type']= construction.css('dd.listing-features__description--dwelling_type span.listing-features__main-description::text').get()
        house_item['Property Type']= construction.css('dd.listing-features__description--property_types span.listing-features__main-description::text').get()
        house_item['Construction Type']= construction.css('dd.listing-features__description--construction_type span.listing-features__main-description::text').get()
        house_item['Year of Construction']= construction.css('dd.listing-features__description--construction_period span.listing-features__main-description::text').get() # Change to integer
        house_item['Number of Rooms']= layout.css('dd.listing-features__description--number_of_rooms span.listing-features__main-description::text').get()
        house_item['Number of Bedrooms']= layout.css('dd.listing-features__description--number_of_bedrooms span.listing-features__main-description::text').get() # Change to integer
        house_item['Number of Bathrooms']= layout.css('dd.listing-features__description--number_of_bathrooms span.listing-features__main-description::text').get() # Change to integer
        house_item['Balcony']= outdoor.css('dd.listing-features__description--balcony span.listing-features__main-description::text').get()
        house_item['Garden']= outdoor.css('dd.listing-features__description--garden span.listing-features__main-description::text').get()
        house_item['Energy Rating']= energy.css('dt:contains("Energy rating") + dd span.listing-features__main-description::text').get()
        house_item['Smoking Allowed']= condition.css('dd.listing-features__description--smoking_allowed span.listing-features__main-description::text').get()
        house_item['Pets Allowed']= condition.css('dd.listing-features__description--pets_allowed span.listing-features__main-description::text').get()
        house_item['Offered Since']= transfer.css('dd.listing-features__description--offered_since span.listing-features__main-description::text').get() # Change to date
        house_item['Status']= transfer.css('dd.listing-features__description--status span.listing-features__main-description::text').get()
        house_item['Available']= transfer.css('dd.listing-features__description--acceptance span.listing-features__main-description::text').get() # change to date
        house_item['Rental Agreement']= transfer.css('dd.listing-features__description--contract_duration span.listing-features__main-description::text').get()
        house_item['Duration']=  transfer.css('dd.listing-features__description--contract_duration_min_max span.listing-features__main-description::text').get()
        house_item['Agent Name']= agent.css('a.agent-summary__title-link::text').get()
        house_item['Agent URL']= agent_page
        house_item['House URL']= response.css("link[rel='alternate'][hreflang='en']::attr(href)").get()
        house_item['Description']= response.css('div.listing-detail-description__additional *::text').getall()
        yield house_item


