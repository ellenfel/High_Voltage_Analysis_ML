#!/bin/bash

# Database credentials
DB_NAME="HV_analysis_ds"
DB_USER="postgres"
DB_PASS="postgres"
OUTPUT_FILE="/home/ellenfel/Desktop/repos/data/hv.csv"

# Setting environment variables for PostgreSQL
export PGPASSWORD=$DB_PASS
export PGHOST=localhost
export PGUSER=$DB_USER
export PGDATABASE=$DB_NAME

echo "=== Database Export Script ==="
echo "Starting time: $(date)"

# Drop tables if they exist (with timing)
echo "Dropping existing tables..."
time psql -c "
DROP TABLE IF EXISTS veri;
DROP TABLE IF EXISTS last_veri;
"

# Get total record count (with better counting method)
echo "Counting total records..."
TOTAL_RECORDS=$(psql -t -c "
    SELECT COUNT(*)
    FROM ts_kv
    JOIN device ON device.Id = ts_kv.entity_id
    JOIN device_profile ON device.device_profile_id = device_profile.id
    WHERE device_profile.name = 'I-Link Box'
")
echo "Total records to process: $TOTAL_RECORDS"

# Create table with direct insert - much faster approach
echo "Creating and populating veri table directly (this may take a while)..."
time psql -c "
CREATE TABLE veri AS (
    SELECT
        device.Id,
        device.name AS devName,
        ts_kv.ts,
        ts_kv.key AS telemetry,
        CONCAT(
            COALESCE(CAST(ts_kv.bool_v AS TEXT), ''),
            ' ',
            COALESCE(CAST(ts_kv.long_v AS TEXT), ''),
            ' ',
            COALESCE(CAST(ts_kv.dbl_v AS TEXT), ''),
            ' ',
            COALESCE(ts_kv.str_v, '')
        ) AS merged_column
    FROM ts_kv
    JOIN device ON device.Id = ts_kv.entity_id
    JOIN device_profile ON device.device_profile_id = device_profile.id
    WHERE device_profile.name != 'default' -- Exclude the 'default' device profile
);

-- Create indexes after data is loaded (much faster than during insertion)
CREATE INDEX idx_veri_telemetry ON veri(telemetry);
"

echo "Creating last_veri table with key dictionary join..."
time psql -c "
CREATE TABLE last_veri AS (
    SELECT veri.*, key_dictionary.key AS key
    FROM veri
    JOIN key_dictionary ON veri.telemetry = key_dictionary.key_id
);
"

echo "Exporting data to CSV..."
time psql -c "
COPY (
    SELECT
        to_timestamp(ts / 1000) AS ts,
        devName,
        key,
        merged_column
    FROM last_veri
    ORDER BY ts
) TO STDOUT WITH CSV HEADER
" > "$OUTPUT_FILE"

echo "Done! Data exported to $OUTPUT_FILE"
echo "Final file size: $(du -h $OUTPUT_FILE)"
echo "Ending time: $(date)"

# Clean up
echo "Cleaning up temporary tables..."
time psql -c "
DROP TABLE IF EXISTS veri;
DROP TABLE IF EXISTS last_veri;
"

echo "Script completed successfully."