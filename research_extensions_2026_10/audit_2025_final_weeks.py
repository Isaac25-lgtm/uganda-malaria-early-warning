"""Check the retained December evaluation week against the current source vintage."""
from research_extensions_2026_10.audit_2020_source import main


if __name__=='__main__':
    main(periods=[f'2025W{w}' for w in range(49,53)],folder='dhis2_2025_final_weeks_audit',report_name='2025_final_weeks_live_audit.json')
