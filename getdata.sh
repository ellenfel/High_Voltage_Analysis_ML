#!/bin/bash

# Database credentials
DB_NAME="highvoltage"
DB_USER="postgres"
DB_PASS="postgres"

# SQL query
SQL=" 

DROP TABLE IF EXISTS veri;
DROP TABLE IF EXISTS last_veri;

-- Create the veri table with the device type filter
CREATE TABLE veri AS (
    SELECT
        device.Id,
        device.name AS devName,
        ts_kv.ts,
        ts_kv.key,
        ts_kv.long_v,
        ts_kv.dbl_v,
        ts_kv.str_v,
        ts_kv.bool_v
    FROM ts_kv
    JOIN device ON device.Id = ts_kv.entity_id
    JOIN device_profile ON device.device_profile_id = device_profile.id
--    WHERE random() < 0.01
    AND device_profile.name = 'I-Link Box' -- exclude 'UG-67' Device and only use specified profile
);

ALTER TABLE veri ADD COLUMN merged_column varchar(512); -- Adjust the length as needed

UPDATE veri SET merged_column = CONCAT(bool_v, ' ', long_v, ' ', dbl_v, ' ', str_v);

ALTER TABLE veri DROP COLUMN long_v, DROP COLUMN dbl_v, DROP COLUMN str_v, DROP COLUMN bool_v;

ALTER TABLE veri RENAME COLUMN key TO telemetry;

CREATE TABLE last_veri AS (
    SELECT veri.*, key_dictionary.key AS key
    FROM veri
    JOIN key_dictionary ON veri.telemetry = key_dictionary.key_id
);


-- Select Relevant Columns from last_veri Table with Filtering and Formatting
SELECT
    to_timestamp(ts / 1000) AS ts,
    devName,
    key,
    merged_column
FROM last_veri


"


# Execute SQL query and export the result to CSV
export PGPASSWORD=$DB_PASS
echo "$SQL" | psql -h localhost -U $DB_USER -d $DB_NAME -a -o /home/ellenfel/Desktop/repos/data/hv.csv
echo "hey"

# Now, you can convert the CSV to XLS if desired. For example, using `ssconvert` (from Gnumeric package):
# ssconvert veri.csv veri.xls