"""Offline descriptive census. Run only after a frozen journal is available.

Imports and pure functions do not write files. main() validates the frozen inputs
and journal before creating analysis/metrics.json and analysis/per_cell.csv.
No runner/transport imports, inference, resampling, or population inference.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import product
import json
from pathlib import Path
import re

import build_plan as plan

ORIENTATIONS = ('forward', 'reverse')
CONDITIONS = ('labels', 'definitions')
ARMS = ('D', 'F', 'L')
MIRROR = dict(zip(plan.LABELS, ('B_SUB_A', 'A_SUB_B', 'EQUIVALENT', 'OTHER', 'UNRESOLVED')))
STATUSES = ('ok', 'invalid_answer', 'incomplete_answer', 'http_error',
            'unknown_usage', 'usage_bound_exceeded', 'transport_error')
PRIOR_CNY = plan.PAST


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def constant(_):
        raise ValueError('nonfinite JSON number')

    def finite_float(text):
        value = float(text)
        require(-float('inf') < value < float('inf'), 'nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def money(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('invalid cost') from exc
    require(result.is_finite() and result >= 0, 'invalid cost')
    return result


def reference(orientation):
    require(orientation in ORIENTATIONS, 'invalid orientation')
    return 'A_SUB_B' if orientation == 'forward' else 'B_SUB_A'


def materials(edge, orientation, definitions=True):
    reference(orientation)
    roles = ('child', 'parent') if orientation == 'forward' else ('parent', 'child')
    return {key: {'label': edge[role + '_label'],
                  **({'definitions': edge[role + '_definition']} if definitions else {})}
            for key, role in zip(('A', 'B'), roles)}


def graph_components(edges):
    """Weak components of the observed endpoint graph; stable lexical IDs."""
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for edge in edges:
        a, b = find(edge['child_id']), find(edge['parent_id'])
        parents[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for node in sorted(parents):
        groups[find(node)].append(node)
    components, membership = [], {}
    for index, nodes in enumerate(sorted(groups.values()), 1):
        node_set = set(nodes)
        ids = sorted(e['id'] for e in edges if e['child_id'] in node_set)
        name = 'component_%03d' % index
        components.append({'component': name, 'nodes': nodes, 'edge_ids': ids,
                           'node_count': len(nodes), 'edge_count': len(ids)})
        membership.update(dict.fromkeys(ids, name))
    return components, membership


def validate_inputs(edges, rows, full_census=True):
    """Validate the design grid; small balanced fixtures may opt out of census sizes.

    Source OWL/provenance verification belongs to the parent's preflight. This
    checks the frozen extracted assertions and never invents negative references.
    """
    require(type(edges) is list and bool(edges), 'empty edge inventory')
    edge_map, pairs = {}, set()
    for edge in edges:
        require(type(edge) is dict and type(edge.get('id')) is str, 'invalid edge')
        require(edge['id'] not in edge_map, 'duplicate edge ID')
        require(edge.get('source_assertion') == 'rdfs:subClassOf', 'non-source assertion')
        for role in ('child', 'parent'):
            require(all(type(edge.get(role + suffix)) is str and edge[role + suffix]
                        for suffix in ('_id', '_label')), 'invalid endpoint')
            defs = edge.get(role + '_definition')
            require(type(defs) is list and bool(defs) and
                    all(type(d) is str and bool(d) for d in defs), 'invalid definitions')
        pair = (edge['child_id'], edge['parent_id'])
        require(pair[0] != pair[1] and pair not in pairs and pair[::-1] not in pairs,
                'duplicate/self/opposing edge')
        pairs.add(pair)
        require(type(edge.get('definition_names_parent')) is bool, 'invalid cue flag')
        families = edge.get('families')
        require(type(families) is list and bool(families) and
                all(type(f) is str and bool(f) for f in families) and
                len(families) == len(set(families)), 'invalid family membership')
        edge_map[edge['id']] = edge
    require(type(rows) is list, 'invalid plan')
    by_id, grid, smoke = {}, set(), set()
    for row in rows:
        require(type(row) is dict, 'invalid plan row')
        phase, edge_id, orientation, condition, arm, provider = (
            row.get(k) for k in ('phase', 'edge_id', 'orientation', 'condition', 'arm', 'provider'))
        require(phase in ('formal', 'smoke') and orientation in ORIENTATIONS and
                condition in CONDITIONS and arm in ARMS and provider in plan.PROVIDERS,
                'incompatible plan dimensions')
        logical = '/'.join((phase, edge_id, orientation, condition, arm, provider))
        require(row.get('logical_id') == logical and logical not in by_id, 'duplicate/invalid logical ID')
        require(row.get('reference') == reference(orientation), 'incompatible reference direction')
        require(row.get('request_sha256') == plan.digest(plan.canonical(row['payload']).encode('utf-8')),
                'request hash mismatch')
        money(row['reserved_cny'])
        if phase == 'formal':
            require(edge_id in edge_map, 'unknown formal edge')
            require(row == plan.make_request(edge_map[edge_id], orientation, condition, arm, provider, phase),
                    'formal plan/material binding mismatch')
            grid.add((edge_id, orientation, condition, arm, provider))
        else:
            require(edge_id == 'TOY_ONLY' and condition == 'definitions', 'invalid smoke cell')
            smoke.add((orientation, arm, provider))
        by_id[logical] = row
    expected = set(product(edge_map, ORIENTATIONS, CONDITIONS, ARMS, plan.PROVIDERS))
    require(grid == expected, 'incomplete formal plan grid')
    if full_census:
        require(len(edges) == 93 and len(grid) == 2232, 'wrong census size')
        require(smoke == set(product(ORIENTATIONS, ARMS, plan.PROVIDERS)), 'incomplete smoke plan')
        require(sum(e['definition_names_parent'] for e in edges) == 85, 'wrong cue strata')
        require(len({f for e in edges for f in e['families']}) == 7, 'wrong family inventory')
        require(len({n for pair in pairs for n in pair}) == 98, 'wrong endpoint inventory')
    return by_id


def usage_error(usage, row):
    """Same accounting schema as public_runner, without importing transport."""
    if type(usage) is not dict:
        return 'unknown_usage'
    for name, bound in (('prompt_tokens', row['input_bound']), ('completion_tokens', row['output_bound'])):
        if type(usage.get(name)) is not int or usage[name] < 0:
            return 'unknown_usage'
    if 'total_tokens' in usage and (type(usage['total_tokens']) is not int or
            usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']):
        return 'unknown_usage'
    if usage['prompt_tokens'] > row['input_bound'] or usage['completion_tokens'] > row['output_bound']:
        return 'usage_bound_exceeded'

    def counters(value, bound):
        return (all(counters(v, bound) for v in value.values()) if type(value) is dict
                else type(value) is int and 0 <= value <= bound)
    for name, value in usage.items():
        if name not in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if not counters(value, row['input_bound'] if name.startswith('prompt') else row['output_bound']):
                return 'unknown_usage'
    return None


def inspect_finish(row, event):
    """Recompute derived journal fields from retained raw response, purely offline."""
    result = dict(response=None, response_model=None, usage=None, label=None,
                  finish_reason=None, estimated_cost_cny=row['reserved_cny'])
    if event.get('status') == 'transport_error':
        require(event.get('rawbody') is None and event.get('http_status') is None,
                'contradictory transport-error response')
        return dict(result, status='transport_error')
    try:
        response = strict_json(event.get('rawbody'))
    except (ValueError, TypeError, RecursionError):
        response = None
    result['response'] = response
    if type(response) is not dict:
        return dict(result, status='unknown_usage')
    result.update(usage=response.get('usage'), response_model=(
        response.get('model') if isinstance(response.get('model'), str) else None))
    error = usage_error(result['usage'], row)
    charge = None
    if error in (None, 'usage_bound_exceeded'):
        usage, rates = result['usage'], plan.PROVIDERS[row['provider']]
        charge = (Decimal(rates['input_rate']) * usage['prompt_tokens'] +
                  Decimal(rates['output_rate']) * usage['completion_tokens']) / Decimal(1000000)
    if error:
        if error == 'usage_bound_exceeded':
            result['estimated_cost_cny'] = format(max(money(row['reserved_cny']), charge), 'f')
        return dict(result, status=error)
    result['estimated_cost_cny'] = format(charge, 'f')
    choices = response.get('choices')
    choice = choices[0] if type(choices) is list and len(choices) == 1 and type(choices[0]) is dict else {}
    result['finish_reason'] = choice.get('finish_reason')
    message = choice.get('message')
    label = None
    if type(message) is dict and type(message.get('content')) is str:
        try:
            obj = strict_json(message['content'])
            if type(obj) is dict and set(obj) == {'label'} and type(obj['label']) is str and obj['label'] in plan.LABELS:
                label = obj['label']
        except (ValueError, TypeError, RecursionError):
            pass
    http = event.get('http_status')
    if type(http) is not int or not 200 <= http < 300:
        status = 'http_error'
    elif result['finish_reason'] != 'stop':
        status = 'incomplete_answer'
    elif type(message) is not dict or message.get('refusal') or label is None:
        status = 'invalid_answer'
    else:
        status, result['label'] = 'ok', label
    return dict(result, status=status)


def validate_journal(rows, events, binding):
    require(type(binding) is str and re.fullmatch('[0-9a-f]{64}', binding), 'invalid freeze binding')
    by_id = {r['logical_id']: r for r in rows}
    require(len(by_id) == len(rows), 'duplicate plan ID')
    starts, finishes, attempts = {}, {}, set()
    for event in events:
        require(type(event) is dict and event.get('freeze_sha256') == binding, 'journal freeze mismatch')
        logical = event.get('logical_id')
        require(logical in by_id, 'unknown journal request')
        row = by_id[logical]
        require(all(event.get(k) == row[k] for k in ('request_sha256', 'reserved_cny')),
                'journal request binding mismatch')
        # Finishes need not repeat dimensions, but any repeated fields must agree.
        require(all(event[k] == value for k, value in row.items() if k in event), 'journal plan mismatch')
        if event.get('type') == 'start':
            require(logical not in starts, 'duplicate start')
            require(all(event.get(k) == v for k, v in row.items()), 'invalid start plan')
            attempt = event.get('attempt_id')
            require(type(attempt) is str and re.fullmatch('[0-9a-f]{32}', attempt) and
                    attempt not in attempts and type(event.get('start_utc')) is str, 'invalid/duplicate attempt')
            starts[logical] = event
            attempts.add(attempt)
        elif event.get('type') == 'finish':
            require(logical in starts and logical not in finishes, 'orphan/duplicate finish')
            require(all(event.get(k) == starts[logical][k] for k in ('attempt_id', 'start_utc')),
                    'finish attempt mismatch')
            expected = inspect_finish(row, event)
            require(all(k in event and event[k] == v for k, v in expected.items()), 'journal result mismatch')
            charge = money(event['estimated_cost_cny'])
            require(event['status'] == 'usage_bound_exceeded' or charge <= money(row['reserved_cny']),
                    'cost exceeds reservation without validated over-bound usage')
            finishes[logical] = event
        else:
            raise ValueError('invalid journal event type')
    return starts, finishes


def per_cell(edges, rows, starts, finishes):
    edge_map = {e['id']: e for e in edges}
    _, membership = graph_components(edges)
    cells = []
    for row in sorted(rows, key=lambda r: r['logical_id']):
        if row['phase'] != 'formal':
            continue
        edge, logical = edge_map[row['edge_id']], row['logical_id']
        event = finishes.get(logical)
        status = event['status'] if event else ('pending' if logical in starts else 'unattempted')
        valid = status == 'ok'
        label = event['label'] if valid else None
        forward = row['orientation'] == 'forward'
        material = materials(edge, row['orientation'], row['condition'] == 'definitions')
        cells.append({**{k: row[k] for k in ('logical_id', 'edge_id', 'provider', 'condition', 'arm', 'orientation')},
                      'requested_model': row['payload']['model'], 'A_id': edge['child_id' if forward else 'parent_id'],
                      'B_id': edge['parent_id' if forward else 'child_id'],
                      'source_child_id': edge['child_id'], 'source_parent_id': edge['parent_id'],
                      'source_assertion': edge['source_assertion'], 'reference': reference(row['orientation']),
                      'component': membership[edge['id']], 'families': edge['families'],
                      'definition_names_parent': edge['definition_names_parent'], 'status': status,
                      'attempted': logical in starts, 'finished': event is not None,
                      'returned': event is not None and type(event.get('rawbody')) is str,
                      'valid_returned': valid, 'predicted_label': label,
                      'match': valid and label == reference(row['orientation']),
                      'response_model': event.get('response_model') if event else None,
                      'constant_A_label': 'A_SUB_B', 'name_containment_label': plan.lexical_label(material),
                      'definition_containment_label': plan.lexical_label(material, True)
                      if row['condition'] == 'definitions' else None})
    return cells


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(cells):
    n, valid = len(cells), sum(c['valid_returned'] for c in cells)
    matches = sum(c['match'] for c in cells)
    pairs = defaultdict(dict)
    for cell in cells:
        key = tuple(cell[k] for k in ('edge_id', 'provider', 'condition', 'arm'))
        require(cell['orientation'] not in pairs[key], 'duplicate analysis cell')
        pairs[key][cell['orientation']] = cell
    require(all(set(p) == set(ORIENTATIONS) for p in pairs.values()), 'unbalanced analysis orientations')
    both_valid = both_correct = consistent = 0
    for pair in pairs.values():
        a, b = pair['forward'], pair['reverse']
        both_correct += a['match'] and b['match']
        both_valid += a['valid_returned'] and b['valid_returned']
        consistent += bool(a['valid_returned'] and b['valid_returned'] and
                           MIRROR[a['predicted_label']] == b['predicted_label'])
    statuses = Counter(c['status'] for c in cells)
    labels = Counter(c['predicted_label'] for c in cells if c['valid_returned'])
    return {'scheduled': n, 'attempted': sum(c['attempted'] for c in cells),
            'finished': sum(c['finished'] for c in cells), 'returned': sum(c['returned'] for c in cells),
            'valid_returned': valid, 'invalid_returned': sum(c['returned'] and not c['valid_returned'] for c in cells),
            'pending': statuses['pending'], 'unattempted': statuses['unattempted'],
            'matches': matches, 'nonmatches': n - matches, 'agreement': fraction(matches, n),
            'returned_valid_agreement': fraction(matches, valid),
            'label_counts': {label: labels[label] for label in plan.LABELS}, 'no_valid_label': n - valid,
            'status_counts': {s: statuses[s] for s in (*STATUSES, 'pending', 'unattempted')},
            'edge_presentations': len(pairs), 'both_orientations_valid': both_valid,
            'both_orientations_correct': both_correct, 'both_orientations_correct_rate': fraction(both_correct, len(pairs)),
            'reversal_consistent': consistent, 'reversal_consistency': fraction(consistent, len(pairs)),
            'reversal_consistency_both_valid': fraction(consistent, both_valid)}


def tables(cells):
    primary, comparisons = [], []
    for provider, condition in product(sorted(plan.PROVIDERS), CONDITIONS):
        arms = {}
        for arm in ARMS:
            subset = [c for c in cells if (c['provider'], c['condition'], c['arm']) == (provider, condition, arm)]
            arms[arm] = summarize(subset)
            primary.append(dict(provider=provider, condition=condition, arm=arm, **arms[arm]))
        row = {'provider': provider, 'condition': condition, 'arms': arms}
        for other in ('L', 'D'):
            f, b = arms['F'], arms[other]
            row['F_minus_' + other + '_pp'] = (100 * (f['matches'] - b['matches']) / f['scheduled']
                                               if f['scheduled'] else None)
        comparisons.append(row)
    return primary, comparisons


def baseline_tables(cells):
    result = []
    # Baselines depend on materials, not model or arm: count each orientation once.
    provider = sorted(plan.PROVIDERS)[0]
    for condition in CONDITIONS:
        subset = [c for c in cells if c['provider'] == provider and c['arm'] == 'D' and c['condition'] == condition]
        for name in ('constant_A', 'name_containment', 'definition_containment'):
            if name == 'definition_containment' and condition != 'definitions':
                continue
            predicted = [{**c, 'predicted_label': c[name + '_label'],
                          'match': c[name + '_label'] == c['reference'],
                          'status': 'ok', 'valid_returned': True, 'returned': True,
                          'attempted': True, 'finished': True} for c in subset]
            score = summarize(predicted)
            # Baselines are deterministic rules; operational model fields do not apply.
            for key in ('attempted', 'finished', 'returned', 'valid_returned', 'invalid_returned',
                        'pending', 'unattempted', 'returned_valid_agreement', 'status_counts'):
                score.pop(key)
            result.append(dict(baseline=name, condition=condition,
                               abstentions=score['label_counts']['UNRESOLVED'], **score))
    return result


def accounting(rows, starts, finishes):
    def subtotal(selected):
        known, unknown, future = Decimal(0), Decimal(0), Decimal(0)
        tokens, models = Counter(), Counter()
        known_n = unknown_n = over_bound_n = 0
        over_bound_charge = Decimal(0)
        for row in selected:
            logical = row['logical_id']
            event = finishes.get(logical)
            if event and event.get('response_model') is not None:
                models[event['response_model']] += 1
            if logical not in starts:
                future += money(row['reserved_cny'])
            elif event and event['status'] not in ('transport_error', 'unknown_usage'):
                known += money(event['estimated_cost_cny'])
                known_n += 1
                if event['status'] == 'usage_bound_exceeded':
                    over_bound_n += 1
                    over_bound_charge += money(event['estimated_cost_cny'])
                usage = event['usage']
                tokens['prompt_tokens'] += usage['prompt_tokens']
                tokens['completion_tokens'] += usage['completion_tokens']
                tokens['total_tokens'] += usage['prompt_tokens'] + usage['completion_tokens']
            else:
                unknown += money(row['reserved_cny'])
                unknown_n += 1
        return {'scheduled': len(selected), 'known_usage_finishes': known_n,
                'known_over_bound_finishes': over_bound_n,
                'known_over_bound_conservative_journal_cny': str(over_bound_charge),
                'unknown_or_pending_attempts': unknown_n,
                'new_actual_conservative_journal_cny': str(known),
                'unknown_full_reserved_liability_cny': str(unknown),
                'new_conservative_liability_cny': str(known + unknown),
                'unattempted_reservation_cny': str(future),
                'full_suite_reservation_cny': str(sum((money(r['reserved_cny']) for r in selected), Decimal(0))),
                'validated_usage_totals': {k: tokens[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                'returned_model_identifiers': dict(sorted(models.items()))}
    total = subtotal(rows)
    return {**total, 'includes_smoke': True, 'prior_conservative_cny': str(PRIOR_CNY),
            'total_with_prior_cny': str(PRIOR_CNY + money(total['new_conservative_liability_cny'])),
            'by_phase': {p: subtotal([r for r in rows if r['phase'] == p]) for p in ('formal', 'smoke')},
            'by_provider': {p: subtotal([r for r in rows if r['provider'] == p]) for p in sorted(plan.PROVIDERS)},
            'usage_note': 'Totals include validated token totals, including known over-bound usage. Over-bound journal cost is max(reservation, computed cost). Unknown/pending usage is not zero. Costs are conservative journal estimates, not invoices.'}


def analyze(edges, rows, events, binding, full_census=True):
    """Pure entry point: return metrics and formal per-cell records, never write."""
    validate_inputs(edges, rows, full_census)
    starts, finishes = validate_journal(rows, events, binding)
    cells = per_cell(edges, rows, starts, finishes)
    deviations = [{'logical_id': c['logical_id'], 'requested_model': c['requested_model'],
                   'returned_model': c['response_model'], 'reason': 'missing_or_mismatched_model_identifier'}
                  for c in cells if (c['valid_returned'] or c['response_model'] is not None)
                  and c['response_model'] != c['requested_model']]
    primary, comparisons = tables(cells)
    components, _ = graph_components(edges)

    def group(kind, name, subset):
        arms, contrasts = tables(subset)
        return {'kind': kind, 'group': name, 'edge_count': len({c['edge_id'] for c in subset}),
                'primary_cells': arms, 'model_input_rows': contrasts}

    groups = [group('definition_cue', str(cue).lower(),
                    [c for c in cells if c['definition_names_parent'] == cue]) for cue in (True, False)]
    groups += [group('family', f, [c for c in cells if f in c['families']])
               for f in sorted({f for e in edges for f in e['families']})]
    groups += [group('component', g['component'], [c for c in cells if c['component'] == g['component']])
               for g in components]
    leave_out = []
    for component in components:
        remaining = [c for c in cells if c['component'] != component['component']]
        _, contrasts = tables(remaining)
        leave_out.append({'omitted_component': component['component'],
                          'remaining_edges': len(edges) - component['edge_count'],
                          'model_input_rows': [{k: r[k] for k in ('provider', 'condition', 'F_minus_L_pp', 'F_minus_D_pp')}
                                               for r in contrasts]})
    ranges = []
    for row in comparisons:
        item = {k: row[k] for k in ('provider', 'condition')}
        for effect in ('F_minus_L_pp', 'F_minus_D_pp'):
            values = [r[effect] for omission in leave_out for r in omission['model_input_rows']
                      if all(r[k] == row[k] for k in ('provider', 'condition')) and r[effect] is not None]
            item[effect] = {'min': min(values) if values else None, 'max': max(values) if values else None,
                            'defined_omissions': len(values)}
        ranges.append(item)
    return {'schema_version': 1, 'scope': 'Fixed census of explicit BCIO subclass assertions; descriptive source-direction agreement only.',
            'denominator_policy': 'Every scheduled formal cell; invalid, pending and unattempted are nonmatches. Smoke excluded.',
            'consistency_policy': 'A_SUB_B/B_SUB_A swap; EQUIVALENT/OTHER/UNRESOLVED self-mirror. Consistency is not correctness.',
            'inference_policy': 'One call per cell; no bootstrap, confidence interval, p-value or repeatability estimate.',
            'family_policy': 'Memberships overlap; family counts must not be added as independent samples.',
            'freeze_sha256': binding, 'source_edges': len(edges), 'scientific_scheduled': len(cells),
            'within_model_attribution_valid': not deviations,
            'model_identity_deviations': deviations,
            'model_identity_policy': 'Exact requested identifier required; no provider aliases approved. Deviations retain all cells and costs, but block unqualified within-model conclusions.',
            'formal_journal_complete': all(c['finished'] for c in cells),
            'formal_all_valid': all(c['valid_returned'] for c in cells),
            'overall': summarize(cells), 'primary_cells': primary, 'model_input_rows': comparisons,
            'baselines': baseline_tables(cells), 'strata': groups, 'components': components,
            'leave_one_component_out': leave_out, 'leave_one_component_out_effect_ranges': ranges,
            'accounting': accounting(rows, starts, finishes)}, cells


def load_inputs(root):
    """Read one journal snapshot; no gate changes, recovery, retry, or output."""
    names = ('data/bcio_edges.json', 'planning/requests.provisional.json', 'build_plan.py')
    captured = {name: (root / name).read_bytes() for name in names}
    manifest_raw = (root / 'freeze_manifest.json').read_bytes()
    manifest = strict_json(manifest_raw)
    for name, raw in captured.items():
        require(manifest.get(name) == hashlib.sha256(raw).hexdigest(), 'frozen analysis input hash mismatch')
    require(captured['build_plan.py'] == Path(plan.__file__).read_bytes(), 'loaded generator differs from frozen generator')
    raw_events = (root / 'results/events.jsonl').read_bytes()
    require(not raw_events or raw_events.endswith(b'\n'), 'torn journal')
    events = [strict_json(line) for line in raw_events.splitlines()]
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in captured.items()}
    hashes['results/events.jsonl'] = hashlib.sha256(raw_events).hexdigest()
    return (strict_json(captured[names[0]]), strict_json(captured[names[1]]), events,
            hashlib.sha256(manifest_raw).hexdigest(), hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    edges, rows, events, binding, hashes = load_inputs(args.root)
    metrics, cells = analyze(edges, rows, events, binding)
    metrics['input_sha256'] = hashes
    output = args.root / 'analysis'
    output.mkdir(exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'per_cell.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]), lineterminator='\n')
        writer.writeheader()
        for cell in cells:
            writer.writerow({**cell, 'families': json.dumps(cell['families'], ensure_ascii=False)})
    return metrics


if __name__ == '__main__':
    main()
