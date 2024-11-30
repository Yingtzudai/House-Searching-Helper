# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


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
                    # Remove non-numeric characters like '€' and ','
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
                    value = int(value)
                except ValueError:
                    adapter[float_key] = None

        ## Change to date (available)

        ## Change to date (offered since)

        return item
