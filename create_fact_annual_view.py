# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 22:29:29 2025

@author: Pu Chen
"""

# create_fact_annual_view.py
import duckdb
from pathlib import Path

def create_fact_annual_view():
    deploy_root = Path(__file__).parent
    db_path = deploy_root / "warehouse" / "db" / "global.duckdb"
    
    print(f"📁 Database: {db_path}")
    print(f"📊 fact_annual.parquet exists: {(deploy_root / 'warehouse' / 'facts' / 'fact_annual' / 'fact_annual.parquet').exists()}")
    
    conn = duckdb.connect(str(db_path))
    
    # Check current views
    views = conn.execute("SELECT view_name FROM duckdb_views()").df()
    print("📋 Current views:", views['view_name'].tolist())
    
    # Create the missing view
    print("🔄 Creating fact_annual_v...")
    conn.execute("DROP VIEW IF EXISTS fact_annual_v")
    conn.execute("""
        CREATE VIEW fact_annual_v AS 
        SELECT * 
        FROM read_parquet('warehouse/facts/fact_annual/fact_annual.parquet')
    """)
    
    # Verify
    count = conn.execute("SELECT COUNT(*) FROM fact_annual_v").fetchone()[0]
    print(f"✅ fact_annual_v created with {count:,} rows")
    
    conn.close()

if __name__ == "__main__":
    create_fact_annual_view()