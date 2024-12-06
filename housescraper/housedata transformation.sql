use house;

-- Create agent table
CREATE TABLE IF NOT EXISTS agent (
agent_id INT AUTO_INCREMENT PRIMARY KEY,
agent_name VARCHAR(255) NOT NULL,
agent_url VARCHAR(255) NOT NULL UNIQUE
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

-- Create date table
