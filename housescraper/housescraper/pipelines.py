# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from datetime import datetime, timedelta
import pandas as pd


class HousescraperPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        ## Strip if not None and not description 
        field_names = adapter.field_names()
        for field_name in field_names:
            if field_name != 'description':
                value = adapter.get(field_name)
                if value is not None and type(value) == str:
                    adapter[field_name] = value.strip()


        ## Remove Euros sign and comma (Price and deposite)
        price_keys = ['price', 'deposit']
        for price_key in price_keys:
            value = adapter.get(price_key)
            if value is not None:
                try:
                    # Remove non-numeric characters '€' and ','
                    value = value.replace('€', '').replace(',', '')
                    adapter[price_key] = float(value)  # Try converting to float
                except ValueError:
                    # If conversion fails, set the value to None
                    adapter[price_key] = None
        
        ## Extract city from house_name and remove city from house_name
        full_name = adapter.get('house_name')
        if full_name is not None:
            full_name_lis = full_name.split(' in ')
            name = full_name_lis[0]
            city = full_name_lis[1]
            adapter['house_name'] = name
            adapter['city'] = city
        
        ## Extract district from address
        full_address = adapter.get('district')
        district = full_address.split('(')[1]
        district = district.replace(')','')
        adapter['district'] = district


        
        ## Change to float (Living area)
        area_str = adapter.get('living_area_m2')
        if area_str is not None:
            area = area_str.split(' ')[0]
            area = float(area)
            adapter['living_area_m2'] = area


        ## Change from float to integer (year of construction, number of bedrooms, number of bathrooms)
        float_keys = ['year_of_construction', 'number_of_bedrooms', 'number_of_bathrooms']
        for float_key in float_keys:
            value = adapter.get(float_key)
            if value is not None:
                try:
                    value_int = int(value)
                    adapter[float_key] = value_int
                except ValueError:
                    adapter[float_key] = None

        ## Change to date (available)
        available_str = adapter.get('available')
        if available_str is not None:
            if 'From' in available_str:
                str_date = available_str.split(' ')[1]
                # date = datetime.strptime(str_date,'%d-%m-%Y')
                adapter['available'] = str_date
            if available_str == 'Immediately':
                today = datetime.today().strftime('%d-%m-%Y')
                # today = datetime.strptime(today,'%d-%m-%Y')
                adapter['available'] = today
            if available_str == 'In consultation':
                adapter['available'] = None
            
    

        ## Change to date (offered since)
        offer_str = adapter.get('offered_since')
        if offer_str is not None:
            if 'weeks' not in offer_str and 'months' not in offer_str:
                # past_date = datetime.strptime(offer_str, '%d-%m-%Y')
                past_date = offer_str
            elif 'weeks' in offer_str:
                week_num = int(offer_str.split(' ')[0])
                today = datetime.now()
                past_date = today - timedelta(weeks = week_num)
                past_date = past_date.strftime('%d-%m-%Y')
                # past_date = datetime.strptime(past_date,'%d-%m-%Y')
            elif 'months' in offer_str:
                month_num = int(list(offer_str)[0])
                today = datetime.now()
                past_date = today - timedelta(months = month_num)
                past_date = past_date.strftime('%d-%m-%Y')
                # past_date = datetime.strptime(past_date,'%d-%m-%Y')
            adapter['offered_since'] = past_date
        
        ## Transform description to string
        description_list = adapter.get('description')
        description_str = ' '.join(description_list).strip()
        adapter['description'] = description_str

        return item

import mysql.connector

class SaveToMySQLPipeline:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host = 'localhost',
            user = 'root',
            password = '1234',
            database = 'house'
        )

        ## Create cursor, used to execute commands
        self.cur = self.conn.cursor()

        ## Create house table if none exists
        self.cur.execute("""
                         CREATE TABLE IF NOT EXISTS house(
                         address VARCHAR(255),
                         agent_name text,
                         agent_url VARCHAR(255),
                         available VARCHAR(255),
                         balcony text,
                         city text,
                         construction_type text,
                         deposit DECIMAL,
                         description text,
                         district text,
                         duration VARCHAR(255),
                         dwelling_type text,
                         energy_rating VARCHAR(255),
                         garden text,
                         house_name text,
                         house_url VARCHAR(255),
                         interior text,
                         living_area_m2 DECIMAL,
                         number_of_bathrooms INTEGER,
                         number_of_bedrooms INTEGER,
                         number_of_rooms INTEGER,
                         offered_since VARCHAR(255),
                         pets_allowed text,
                         price DECIMAL,
                         property_type text,
                         rental_agreement text,
                         service_cost text,
                         smoking_allowed text,
                         status text,
                         year_of_construction INTEGER,
                         PRIMARY KEY (house_url)
                         )
                         """)


    def process_item(self, item, spider):

        ## Define insert statement
        self.cur.execute("""
    INSERT INTO house(
        address,
        agent_name,
        agent_url,
        available,
        balcony,
        city,
        construction_type,
        deposit,
        description,
        district,
        duration,
        dwelling_type,
        energy_rating,
        garden,
        house_name,
        house_url,
        interior,
        living_area_m2,
        number_of_bathrooms,
        number_of_bedrooms,
        number_of_rooms,
        offered_since,
        pets_allowed,
        price,
        property_type,
        rental_agreement,
        service_cost,
        smoking_allowed,
        status,
        year_of_construction
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s
    )
""", (
    item['address'],
    item['agent_name'],
    item['agent_url'],
    item['available'],
    item['balcony'],
    item['city'],
    item['construction_type'],
    item['deposit'],
    item['description'],
    item['district'],
    item['duration'],
    item['dwelling_type'],
    item['energy_rating'],
    item['garden'],
    item['house_name'],
    item['house_url'],
    item['interior'],
    item['living_area_m2'],
    item['number_of_bathrooms'],
    item['number_of_bedrooms'],
    item['number_of_rooms'],
    item['offered_since'],
    item['pets_allowed'],
    item['price'],
    item['property_type'],
    item['rental_agreement'],
    item['service_cost'],
    item['smoking_allowed'],
    item['status'],
    item['year_of_construction']
))

      
        # Execute insert of data into database
        self.conn.commit()
        return item

    def close_spider(self, spider):

        ## Close cursor & connection to database
        self.cur.close()
        self.conn.close()
