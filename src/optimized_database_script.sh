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

# Function to check system resources
check_resources() {
    echo "Current system resource usage:"
    free -h | grep "Mem:"
    echo "CPU usage:"
    top -bn1 | grep "Cpu(s)"
    echo ""
}

# Initial resource check
echo "Initial system status:"
check_resources

# Create temporary tables with indexes - this helps with performance
psql -c "
-- Drop tables if they exist
DROP TABLE IF EXISTS veri;
DROP TABLE IF EXISTS last_veri;

-- Create the veri table with the device type filter
CREATE TABLE veri (
    Id UUID,
    devName VARCHAR(255),
    ts BIGINT,
    telemetry VARCHAR(255),
    merged_column VARCHAR(512)
);

-- Create indexes to improve performance
CREATE INDEX idx_veri_id ON veri(Id);
CREATE INDEX idx_veri_telemetry ON veri(telemetry);
CREATE INDEX idx_veri_ts ON veri(ts);
"

echo "Empty tables created. Adding batch processing..."

# Batch size - adjust based on your system capabilities
BATCH_SIZE=100000

# Get total number of relevant records to process
TOTAL_RECORDS=$(psql -t -c "
    SELECT COUNT(*)
    FROM ts_kv
    JOIN device ON device.Id = ts_kv.entity_id
    JOIN device_profile ON device.device_profile_id = device_profile.id
    WHERE device_profile.name = 'I-Link Box'
")

echo "Total records to process: $TOTAL_RECORDS"

# Process data in batches
OFFSET=0
while true; do
    echo "Processing batch starting at offset $OFFSET..."
    
    # Insert data in batches
    INSERTED=$(psql -t -c "
        INSERT INTO veri (Id, devName, ts, telemetry, merged_column)
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
        WHERE device_profile.name = 'I-Link Box'
        ORDER BY ts_kv.ts
        LIMIT $BATCH_SIZE OFFSET $OFFSET
        RETURNING 1
    " | wc -l)
    
    # If no rows were inserted, we're done
    if [ "$INSERTED" -eq "0" ]; then
        echo "All batches processed."
        break
    fi
    
    # Increment the offset for the next batch
    OFFSET=$((OFFSET + BATCH_SIZE))
    
    # Check resources after each batch
    check_resources
    
    # Optional: Add a sleep to give system time to recover
    sleep 2
done

echo "Creating last_veri table with key dictionary join..."

# Create the last_veri table
psql -c "
CREATE TABLE last_veri AS (
    SELECT veri.*, key_dictionary.key AS key
    FROM veri
    JOIN key_dictionary ON veri.telemetry = key_dictionary.key_id
);

CREATE INDEX idx_last_veri_ts ON last_veri(ts);
"

echo "Exporting data to CSV..."

# Export directly to CSV using COPY command (more efficient than psql -o)
psql -c "
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

# Final resource check
echo "Final system status:"
check_resources

# Optional: Clean up temporary tables to free space
psql -c "
DROP TABLE IF EXISTS veri;
DROP TABLE IF EXISTS last_veri;
"

echo "Temporary tables removed."