"""Render verified descriptive tables; requires complete real-journal analysis."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent


def main():
    m=json.loads((ROOT/'analysis/metrics.json').read_text('utf-8'))
    assert m['formal_journal_complete'] and m['within_model_attribution_valid']
    lines=['# Public BCIO source-direction experiment — complete initial schedule','',
           'Scope: agreement with recorded inclusive subclass directions, not general construct accuracy.','',
           '| Model | Input | D matches/186 | F matches/186 | L matches/186 | F−L pp | F−D pp |',
           '|---|---|---:|---:|---:|---:|---:|']
    claims=[]
    for row in m['model_input_rows']:
        counts=[row['arms'][a]['matches'] for a in ('D','F','L')]
        lines.append('| %s | %s | %s | %s | %s | %+.2f | %+.2f |'%
                     (row['provider'],row['condition'],*counts,row['F_minus_L_pp'],row['F_minus_D_pp']))
        for arm,count in zip(('D','F','L'),counts):
            claims.append({'id':row['provider']+'_'+row['condition']+'_'+arm+'_matches',
                           'value':count,'source':'analysis/metrics.json'})
        for contrast in ('F_minus_L_pp','F_minus_D_pp'):
            claims.append({'id':row['provider']+'_'+row['condition']+'_'+contrast,
                           'value':row[contrast],'source':'analysis/metrics.json'})
    lines += ['', '| Model | Input | Arm | Valid/186 | Both orientations matched/93 | Reversal consistent/93 |',
              '|---|---|---|---:|---:|---:|']
    for row in m['primary_cells']:
        lines.append('| %s | %s | %s | %s | %s | %s |'%tuple(row[k] for k in
                     ('provider','condition','arm','valid_returned','both_orientations_correct','reversal_consistent')))
    lines += ['', '| Baseline | Input | Matches/186 | Abstentions/186 |', '|---|---|---:|---:|']
    for row in m['baselines']:
        lines.append('| %s | %s | %s | %s |'%(row['baseline'],row['condition'],row['matches'],row['abstentions']))
    lines += ['', '| Model | Input | F−L leave-one-component-out range (pp) | F−D range (pp) |',
              '|---|---|---:|---:|']
    for row in m['leave_one_component_out_effect_ranges']:
        a,b=row['F_minus_L_pp'],row['F_minus_D_pp']
        lines.append('| %s | %s | %.2f to %.2f | %.2f to %.2f |'%
                     (row['provider'],row['condition'],a['min'],a['max'],b['min'],b['max']))
    lines += ['', 'These ranges are descriptive omissions, not confidence intervals.','',
              '## Completeness and conservative accounting','',
              'Formal scheduled: '+str(m['scientific_scheduled']),
              'Valid formal returns: '+str(m['overall']['valid_returned']),
              'Formal status counts: '+json.dumps(m['overall']['status_counts'],ensure_ascii=False),
              'All A2 calls (including second technical smoke): '+str(m['accounting']['scheduled']),
              'Prior conservative CNY (historical plus A1 smoke): '+m['accounting']['prior_conservative_cny'],
              'A2 known-usage conservative CNY: '+m['accounting']['new_actual_conservative_journal_cny'],
              'A2 unknown full-reservation CNY: '+m['accounting']['unknown_full_reserved_liability_cny'],
              'Total conservative CNY including all prior calls: '+m['accounting']['total_with_prior_cny'],
              'Token totals: '+json.dumps(m['accounting']['validated_usage_totals']),
              'Returned model identifiers: '+json.dumps(m['accounting']['returned_model_identifiers']),
              '', 'The two technical smoke rounds are excluded from scientific denominators. No failed formal cell was retried.',
              'The retained failed calls remain operational nonmatches, with separate valid-return agreement.','']
    (ROOT/'analysis/summary.md').write_text('\n'.join(lines),'utf-8')
    (ROOT/'.aris/claims.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2)+'\n','utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
