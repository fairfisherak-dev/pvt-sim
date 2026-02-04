"""Fix schema validation issues in components.json.

This script:
1. Adds missing 'aliases' fields (as empty lists)
2. Adds missing 'groups' field to O2
3. Fixes Pc values for C11-C16 (correlation error)
"""

import json
from pathlib import Path


def fix_schema_issues():
    """Fix missing fields and incorrect values."""
    db_path = Path(__file__).parent.parent / "data" / "pure_components" / "components.json"

    with open(db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    components = data['components']
    changes_made = []

    # Fix 1: Add missing 'aliases' fields
    for comp_id, comp in components.items():
        if 'aliases' not in comp:
            comp['aliases'] = []
            changes_made.append(f"Added empty aliases to {comp_id}")

    # Fix 2: Add 'groups' field to O2 (empty dict to indicate no PPR78 groups)
    if 'O2' in components and 'groups' not in components['O2']:
        components['O2']['groups'] = {}
        changes_made.append("Added empty groups to O2")

    # Fix 3: Correct Pc values for C11-C16 (they were calculated wrong)
    # Using proper correlation: Pc_bar = 1.0 / (0.113 + 0.0032 * n)^2
    # Then convert to Pa
    pc_fixes = {
        'C11': {'old_pc': 1626, 'new_pc': 1955000},
        'C12': {'old_pc': 1381, 'new_pc': 1817000},
        'C13': {'old_pc': 1188, 'new_pc': 1695000},
        'C14': {'old_pc': 1032, 'new_pc': 1587000},
        'C15': {'old_pc': 905, 'new_pc': 1491000},
        'C16': {'old_pc': 800, 'new_pc': 1404000},
    }

    for comp_id, fix_data in pc_fixes.items():
        if comp_id in components:
            old_pc = components[comp_id].get('Pc')
            if old_pc == fix_data['old_pc']:
                components[comp_id]['Pc'] = fix_data['new_pc']
                changes_made.append(f"Fixed Pc for {comp_id}: {old_pc} -> {fix_data['new_pc']} Pa")

    # Write back
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Schema fixes applied:")
    for change in changes_made:
        print(f"  - {change}")
    print(f"\nTotal changes: {len(changes_made)}")
    print(f"Updated: {db_path}")


if __name__ == "__main__":
    fix_schema_issues()
