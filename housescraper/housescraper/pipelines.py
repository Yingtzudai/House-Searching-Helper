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
                date = datetime.strptime(str_date,'%d-%m-%Y')
                adapter['available'] = date
            if available_str == 'Immediately':
                today = datetime.today().strftime('%d-%m-%Y')
                today = datetime.strptime(today,'%d-%m-%Y')
                adapter['available'] = today
    

        ## Change to date (offered since)
        offer_str = adapter.get('offered_since')
        if offer_str is not None:
            if 'weeks' not in offer_str and 'months' not in offer_str:
                past_date = datetime.strptime(offer_str, '%d-%m-%Y')
            elif 'weeks' in offer_str:
                week_num = int(offer_str.split(' ')[0])
                today = datetime.now()
                past_date = today - timedelta(weeks = week_num)
                past_date = past_date.strftime('%d-%m-%Y')
                past_date = datetime.strptime(past_date,'%d-%m-%Y')
            elif 'months' in offer_str:
                month_num = int(list(offer_str)[0])
                today = datetime.now()
                past_date = today - timedelta(months = month_num)
                past_date = past_date.strftime('%d-%m-%Y')
                past_date = datetime.strptime(past_date,'%d-%m-%Y')
            adapter['offered_since'] = past_date
            
                
            
            


        return item
