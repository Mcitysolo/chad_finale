import os
import psycopg2

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "aimi_dev"
DB_USER = "aimi"
DB_PASS = os.environ.get("PGPASSWORD")

ARTIST_ID = 1

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
)
cur = conn.cursor()

def pending_exists(action_type: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM next_best_actions
        WHERE artist_id = %s
          AND action_type = %s
          AND status = 'pending'
        LIMIT 1
        """,
        (ARTIST_ID, action_type),
    )
    return cur.fetchone() is not None

def insert_action_from_contacts(
    action_type: str,
    contact_type: str,
    priority_score: int,
    expected_impact: str,
    confidence_score: int,
    reason: str,
) -> int:
    if pending_exists(action_type):
        print(f"SKIP existing pending action: {action_type}")
        return 0

    cur.execute(
        """
        INSERT INTO next_best_actions (
          artist_id,
          action_type,
          priority_score,
          reason,
          inputs_json,
          expected_impact,
          confidence_score,
          status
        )
        SELECT
          %s AS artist_id,
          %s AS action_type,
          %s AS priority_score,
          %s AS reason,
          jsonb_build_object(
            'source_table', 'contacts',
            'contact_type', %s,
            'contact_count', COUNT(*),
            'email_count', COUNT(*) FILTER (WHERE email IS NOT NULL AND btrim(email) <> '')
          ) AS inputs_json,
          %s AS expected_impact,
          %s AS confidence_score,
          'pending' AS status
        FROM contacts
        WHERE artist_id = %s
          AND type = %s
        HAVING COUNT(*) > 0
        """,
        (
            ARTIST_ID,
            action_type,
            priority_score,
            reason,
            contact_type,
            expected_impact,
            confidence_score,
            ARTIST_ID,
            contact_type,
        ),
    )
    inserted = cur.rowcount
    if inserted:
        print(f"INSERTED action: {action_type}")
    else:
        print(f"NO SOURCE DATA for action: {action_type}")
    return inserted

inserted_total = 0

inserted_total += insert_action_from_contacts(
    action_type="sync_outreach",
    contact_type="sync_licensing",
    priority_score=85,
    expected_impact="high",
    confidence_score=80,
    reason="artist has live sync licensing contact inventory available for actioning",
)

inserted_total += insert_action_from_contacts(
    action_type="booking_outreach",
    contact_type="booking_agent",
    priority_score=72,
    expected_impact="medium",
    confidence_score=74,
    reason="artist has live booking agent inventory available for outreach actioning",
)

conn.commit()
cur.close()
conn.close()

print("DONE")
print(f"Inserted total: {inserted_total}")
