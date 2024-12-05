# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy





class HousescraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    house_name = scrapy.Field()
    address = scrapy.Field()
    price = scrapy.Field()
    deposit = scrapy.Field()
    service_cost = scrapy.Field()
    living_area_m2 = scrapy.Field()
    number_of_rooms = scrapy.Field()
    interior = scrapy.Field()
    dwelling_type = scrapy.Field()
    property_type = scrapy.Field()
    construction_type = scrapy.Field()
    year_of_construction = scrapy.Field()
    number_of_bedrooms = scrapy.Field()
    number_of_bathrooms = scrapy.Field()
    balcony = scrapy.Field()
    garden = scrapy.Field()
    energy_rating = scrapy.Field()
    smoking_allowed = scrapy.Field()
    pets_allowed = scrapy.Field()
    offered_since = scrapy.Field()
    status = scrapy.Field()
    available = scrapy.Field()
    rental_agreement = scrapy.Field()
    duration = scrapy.Field()
    agent_name = scrapy.Field()
    agent_url = scrapy.Field()
    house_url = scrapy.Field()
    description = scrapy.Field()
