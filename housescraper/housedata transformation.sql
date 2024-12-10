use house;
SET SQL_SAFE_UPDATES = 0;
DELETE FROM house
WHERE price IS NULL;

-- Create agent table
CREATE TABLE IF NOT EXISTS agent (
agent_name VARCHAR(255) NOT NULL,
agent_url VARCHAR(255) NOT NULL UNIQUE PRIMARY KEY
);
INSERT INTO agent(agent_url, agent_name)
SELECT DISTINCT agent_url, agent_name
FROM house;

-- Create territory table
CREATE TABLE IF NOT EXISTS territory (
city VARCHAR(255) NOT NULL,
district VARCHAR(255) NOT NULL
);

INSERT INTO territory(city, district)
SELECT DISTINCT city, district
FROM house;

-- Transform territory table
ALTER TABLE territory
ADD COLUMN region VARCHAR(255);
UPDATE territory
SET region = CONCAT(district, ', ', city);
ALTER TABLE territory ADD PRIMARY KEY (region);

-- Create house_info table
CREATE TABLE IF NOT EXISTS house_info (
                         house_url VARCHAR(255) PRIMARY KEY,
                         house_name text,
                         dwelling_type text,
                         -- construction_type text, (Definition unclear, seriously imbalanced)
                         year_of_construction INTEGER,
                         balcony text,
                         garden text,
                         number_of_bathrooms INTEGER,
                         number_of_bedrooms INTEGER,
                         number_of_rooms INTEGER,
                         living_area_m2 DECIMAL,
                         interior text,
                         energy_rating VARCHAR(255),
                         pets_allowed text,
                         smoking_allowed text
);

INSERT INTO house_info
SELECT house_url, house_name, dwelling_type, year_of_construction, balcony, garden,
number_of_bathrooms, number_of_bedrooms, number_of_rooms, living_area_m2, interior, energy_rating, pets_allowed, 
smoking_allowed
FROM house;
-- Transformation for house_info
UPDATE house_info
SET
    balcony = COALESCE(balcony, 'Not present'), 
    garden = COALESCE(garden, 'Not present'),
    smoking_allowed = COALESCE(smoking_allowed, 'No'),
    pets_allowed = COALESCE(pets_allowed, 'No');
UPDATE house_info
SET 
    number_of_bedrooms = COALESCE(number_of_bedrooms, 0),
    number_of_bathrooms = COALESCE(number_of_bathrooms, 0)
WHERE dwelling_type = 'room';

-- Create rental_info table
CREATE TABLE IF NOT EXISTS rental_info (
                         house_url VARCHAR(255),
                         available DATE,
                         offered_since DATE,
                         -- minimum_months INTEGER, (82% NULL)
                         -- maximum_months INTEGER, (90% NULL)
                         -- rental_agreement text, (51% NULL)
                         price DECIMAL,
                         -- deposit DECIMAL, (37% NULL)
                         service_cost text,
                         status text,
                         agent_url VARCHAR(255),
                         region VARCHAR(255),
                         FOREIGN KEY (agent_url) REFERENCES agent(agent_url),
                         FOREIGN KEY (house_url) REFERENCES house_info(house_url),
                         FOREIGN KEY (region) REFERENCES territory(region)
                         
);
INSERT INTO rental_info
SELECT house_url, STR_TO_DATE(offered_since,'%d-%m-%Y') AS offered_since, STR_TO_DATE(available,'%d-%m-%Y') AS available, price,
service_cost, status, agent_url, CONCAT(district, ', ', city) AS region
FROM house;
-- Transform rental_info
UPDATE rental_info
SET service_cost = CASE 
    WHEN service_cost LIKE '%Electricity%' AND service_cost LIKE '%Gas%' 
         AND service_cost LIKE '%Water%' AND service_cost LIKE '%Internet%' THEN 'All Included'
    WHEN service_cost LIKE '%Includes%' THEN 'Partial Included'
    WHEN service_cost LIKE '%Excludes%' THEN 'Not Included'
    ELSE 'Not Specified'
END;
UPDATE rental_info
SET offered_since = current_date()
WHERE offered_since IS NULL;
SET SQL_SAFE_UPDATES = 1;





