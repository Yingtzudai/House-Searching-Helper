use house;

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

-- Transform territory and house tables
SET SQL_SAFE_UPDATES = 0;
ALTER TABLE house
ADD COLUMN region VARCHAR(255);
UPDATE house
SET region = CONCAT(district, ', ', city);

ALTER TABLE territory
ADD COLUMN region VARCHAR(255);
UPDATE territory
SET region = CONCAT(district, ', ', city);
ALTER TABLE territory ADD PRIMARY KEY (region);
SET SQL_SAFE_UPDATES = 1;


-- Create house_info table
CREATE TABLE IF NOT EXISTS house_info (
                         house_url VARCHAR(255) PRIMARY KEY,
                         house_name text,
                         dwelling_type text,
                         construction_type text,
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
SELECT house_url, house_name, dwelling_type, construction_type, year_of_construction, balcony, garden,
number_of_bathrooms, number_of_bedrooms, number_of_rooms, living_area_m2, interior, energy_rating, pets_allowed, 
smoking_allowed
FROM house;

-- Create rental_info table
CREATE TABLE IF NOT EXISTS rental_info (
                         house_url VARCHAR(255),
                         available DATE,
                         offered_since DATE,
                         minimum_months INTEGER,
                         maximum_months INTEGER,
                         rental_agreement text,
                         price DECIMAL,
                         deposit DECIMAL,
                         service_cost text,
                         status text,
                         agent_url VARCHAR(255),
                         region VARCHAR(255),
                         FOREIGN KEY (agent_url) REFERENCES agent(agent_url),
                         FOREIGN KEY (region) REFERENCES territory(region),
                         FOREIGN KEY (house_url) REFERENCES house_info(house_url)
);
INSERT INTO rental_info
SELECT house_url, STR_TO_DATE(offered_since,'%d-%m-%Y') AS offered_since, STR_TO_DATE(available,'%d-%m-%Y') AS available, minimum_months, maximum_months, rental_agreement, price, deposit,
service_cost, status, agent_url, CONCAT(district, ', ', city) AS region
FROM house;








