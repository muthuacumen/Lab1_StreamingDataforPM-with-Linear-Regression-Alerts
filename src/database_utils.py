"""
Database utilities for Predictive Maintenance project.

Provides functions for:
- Connecting to Neon.tech PostgreSQL
- Ingesting training data
- Querying data for model training
- Logging alerts to database
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime


def get_db_config():
    """Return the default database configuration for Neon.tech."""
    return {
        'host': 'ep-polished-snow-ahx3qiod-pooler.c-3.us-east-1.aws.neon.tech',
        'database': 'neondb',
        'user': 'neondb_owner',
        'password': 'npg_JlIENr3i4AbL',
        'port': 5432,
        'sslmode': 'require'
    }


def connect_to_db(db_config=None):
    """
    Establish connection to PostgreSQL database.

    Parameters:
    -----------
    db_config : dict, optional
        Database configuration. Uses default Neon.tech config if None.

    Returns:
    --------
    connection : psycopg2 connection object
    """
    if db_config is None:
        db_config = get_db_config()

    try:
        conn = psycopg2.connect(**db_config)
        print(f"Connected to database: {db_config['database']} @ {db_config['host']}")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise


def create_training_table(conn):
    """
    Create table for storing training data.

    Parameters:
    -----------
    conn : psycopg2 connection
    """
    cursor = conn.cursor()

    create_query = """
    CREATE TABLE IF NOT EXISTS robot_training_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        robot VARCHAR(20),
        axis_1 FLOAT,
        axis_2 FLOAT,
        axis_3 FLOAT,
        axis_4 FLOAT,
        axis_5 FLOAT,
        axis_6 FLOAT,
        axis_7 FLOAT,
        axis_8 FLOAT,
        axis_9 FLOAT,
        axis_10 FLOAT,
        axis_11 FLOAT,
        axis_12 FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    cursor.execute(create_query)

    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_timestamp
        ON robot_training_data(timestamp);
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_robot
        ON robot_training_data(robot);
    """)

    conn.commit()
    cursor.close()
    print("Training data table created/verified")


def clear_training_table(conn):
    """Clear all data from training table."""
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE robot_training_data RESTART IDENTITY;")
    conn.commit()
    cursor.close()
    print("Training table cleared")


def ingest_training_data(conn, csv_path, batch_size=1000):
    """
    Ingest training data from CSV into database.

    Parameters:
    -----------
    conn : psycopg2 connection
    csv_path : str
        Path to the training CSV file
    batch_size : int
        Number of records per batch insert

    Returns:
    --------
    int : Number of records ingested
    """
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} records from CSV")

    cursor = conn.cursor()

    # Prepare data for batch insert (all 12 axes)
    records = []
    for _, row in df.iterrows():
        record = (
            row['timestamp'],
            row['robot'],
            float(row['axis_1']) if 'axis_1' in row and pd.notna(row['axis_1']) else None,
            float(row['axis_2']) if 'axis_2' in row and pd.notna(row['axis_2']) else None,
            float(row['axis_3']) if 'axis_3' in row and pd.notna(row['axis_3']) else None,
            float(row['axis_4']) if 'axis_4' in row and pd.notna(row['axis_4']) else None,
            float(row['axis_5']) if 'axis_5' in row and pd.notna(row['axis_5']) else None,
            float(row['axis_6']) if 'axis_6' in row and pd.notna(row['axis_6']) else None,
            float(row['axis_7']) if 'axis_7' in row and pd.notna(row['axis_7']) else None,
            float(row['axis_8']) if 'axis_8' in row and pd.notna(row['axis_8']) else None,
            float(row['axis_9']) if 'axis_9' in row and pd.notna(row['axis_9']) else None,
            float(row['axis_10']) if 'axis_10' in row and pd.notna(row['axis_10']) else None,
            float(row['axis_11']) if 'axis_11' in row and pd.notna(row['axis_11']) else None,
            float(row['axis_12']) if 'axis_12' in row and pd.notna(row['axis_12']) else None,
        )
        records.append(record)

    # Batch insert (all 12 axes)
    insert_query = """
    INSERT INTO robot_training_data
    (timestamp, robot, axis_1, axis_2, axis_3, axis_4, axis_5, axis_6, axis_7, axis_8,
     axis_9, axis_10, axis_11, axis_12)
    VALUES %s
    """

    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        execute_values(cursor, insert_query, batch)
        conn.commit()
        total_inserted += len(batch)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Inserted {total_inserted:,} / {len(records):,} records")

    cursor.close()
    print(f"Ingestion complete: {total_inserted:,} records")
    return total_inserted


def query_training_data(conn, robot_name=None, limit=None):
    """
    Query training data from database.

    Parameters:
    -----------
    conn : psycopg2 connection
    robot_name : str, optional
        Filter by robot name (e.g., 'Robot A')
    limit : int, optional
        Maximum number of records to return

    Returns:
    --------
    DataFrame : Training data
    """
    query = "SELECT * FROM robot_training_data"
    params = []

    if robot_name:
        query += " WHERE robot = %s"
        params.append(robot_name)

    query += " ORDER BY timestamp"

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, conn, params=params if params else None)
    print(f"Retrieved {len(df):,} records from database")
    return df


def get_training_record_count(conn):
    """Get total number of records in training table."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM robot_training_data")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def create_alerts_table(conn):
    """
    Create table for storing alert/error events.

    Parameters:
    -----------
    conn : psycopg2 connection
    """
    cursor = conn.cursor()

    create_query = """
    CREATE TABLE IF NOT EXISTS pm_alerts (
        id SERIAL PRIMARY KEY,
        event_timestamp TIMESTAMP,
        axis VARCHAR(20),
        event_type VARCHAR(10),
        deviation FLOAT,
        duration_seconds INT,
        actual_value FLOAT,
        predicted_value FLOAT,
        threshold_used FLOAT,
        robot VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    cursor.execute(create_query)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
        ON pm_alerts(event_timestamp);
    """)

    conn.commit()
    cursor.close()
    print("Alerts table created/verified")


def log_alert_to_db(conn, alert_record):
    """
    Log an alert/error event to database.

    Parameters:
    -----------
    conn : psycopg2 connection
    alert_record : dict
        Alert data with keys: timestamp, axis, event_type, deviation,
        duration_seconds, actual_value, predicted_value, threshold_used, robot
    """
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO pm_alerts
    (event_timestamp, axis, event_type, deviation, duration_seconds,
     actual_value, predicted_value, threshold_used, robot)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        alert_record.get('timestamp'),
        alert_record.get('axis'),
        alert_record.get('event_type'),
        alert_record.get('deviation'),
        alert_record.get('duration_seconds'),
        alert_record.get('actual_value'),
        alert_record.get('predicted_value'),
        alert_record.get('threshold_used'),
        alert_record.get('robot', 'Unknown')
    )

    cursor.execute(insert_query, values)
    conn.commit()
    cursor.close()


def query_alerts(conn, event_type=None, limit=100):
    """
    Query logged alerts from database.

    Parameters:
    -----------
    conn : psycopg2 connection
    event_type : str, optional
        Filter by 'ALERT' or 'ERROR'
    limit : int
        Maximum records to return

    Returns:
    --------
    DataFrame : Alert records
    """
    query = "SELECT * FROM pm_alerts"
    params = []

    if event_type:
        query += " WHERE event_type = %s"
        params.append(event_type)

    query += f" ORDER BY event_timestamp DESC LIMIT {limit}"

    df = pd.read_sql(query, conn, params=params if params else None)
    return df


# Test the module
if __name__ == '__main__':
    print("Testing database utilities...")

    conn = connect_to_db()
    create_training_table(conn)
    create_alerts_table(conn)

    count = get_training_record_count(conn)
    print(f"Current training records: {count:,}")

    conn.close()
    print("Database utilities test complete")
