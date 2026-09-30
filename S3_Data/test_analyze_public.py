"""Hand-expected offline tests. All on-disk fixtures live in TemporaryDirectory.

Run: python -B -m unittest -v test_analyze_public
"""
import copy
import csv
from decimal import Decimal
import importlib.util
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_public as a
import build_plan as plan

BINDING = 'a' * 64


def edge(name, child, parent, child_label='red widget', parent_label='widget',
         child_definition='A widget with a red surface.', cue=True, families=None):
    return {'id': name, 'child_id': child, 'parent_id': parent,
            'child_label': child_label, 'parent_label': parent_label,
            'child_definition': [child_definition], 'parent_definition': ['A manufactured object.'],
            'source_assertion': 'rdfs:subClassOf', 'definition_names_parent': cue,
            'families': families or ['家族一']}


def small_edges():
    return [edge('e1', 'n1', 'n2'),
            edge('e2', 'n3', 'n2', child_label='spark', families=['家族一', '家族二']),
            edge('e3', 'n4', 'n5', child_label='pebble', parent_label='stone',
                 child_definition='A small mineral object.', cue=False, families=['家族二'])]


def requests(edges, smoke=False):
    rows = [plan.make_request(e, o, c, arm, p, 'formal') for e in edges
            for o, c, arm, p in product(a.ORIENTATIONS, a.CONDITIONS, a.ARMS, plan.PROVIDERS)]
    if smoke:
        toy = edge('TOY_ONLY', 'toy_child', 'toy_parent')
        rows += [plan.make_request(toy, o, 'definitions', arm, p, 'smoke')
                 for o, arm, p in product(a.ORIENTATIONS, a.ARMS, plan.PROVIDERS)]
    return rows


def start(row, number=1, binding=BINDING):
    return {**row, 'type': 'start', 'attempt_id': '%032x' % number,
            'start_utc': '2026-09-16T00:00:00Z', 'freeze_sha256': binding}


def finish(begin, label='A_SUB_B', status='ok', response_model='returned-test-model'):
    """Construct journal fixtures independently of the analyzer's response parser."""
    response = {'model': response_model, 'usage': {'prompt_tokens': 10, 'completion_tokens': 2,
                                                 'total_tokens': 12},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'label': label})}}]}
    result = {k: begin[k] for k in ('logical_id', 'attempt_id', 'start_utc', 'freeze_sha256',
                                   'request_sha256', 'reserved_cny')}
    result.update(type='finish', end_utc='2026-09-16T00:00:01Z', status=status,
                  http_status=200, response=response, response_model=response_model,
                  usage=response['usage'], label=label if status == 'ok' else None,
                  finish_reason='stop', header_request_id=None,
                  estimated_cost_cny='0.000045' if begin['provider'] == 'qwen' else '0.000058')
    if status == 'invalid_answer':
        response['choices'][0]['message']['content'] = '{"label":"BAD"}'
    elif status == 'incomplete_answer':
        response['choices'][0]['finish_reason'] = result['finish_reason'] = 'length'
    elif status == 'http_error':
        result['http_status'] = 429
    elif status in ('unknown_usage', 'usage_bound_exceeded'):
        result['finish_reason'] = None
        result['estimated_cost_cny'] = begin['reserved_cny']
        if status == 'unknown_usage':
            response.pop('usage')
            result['usage'] = None
        else:
            result['usage']['completion_tokens'] = 129
            result['usage']['total_tokens'] = 139
    elif status == 'transport_error':
        result.update(response=None, response_model=None, usage=None, finish_reason=None,
                      http_status=None, estimated_cost_cny=begin['reserved_cny'])
    result['rawbody'] = json.dumps(result['response']) if result['response'] is not None else None
    return result


def result_for(metrics, provider='qwen', condition='labels', arm='F'):
    return next(r for r in metrics['primary_cells'] if
                (r['provider'], r['condition'], r['arm']) == (provider, condition, arm))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.edges = small_edges()
        self.rows = requests(self.edges)

    def analyze(self, events=(), rows=None):
        return a.analyze(self.edges, self.rows if rows is None else rows, events, BINDING, full_census=False)

    def select(self, edge_id='e1', orientation='forward', arm='F', condition='labels', provider='qwen'):
        return next(r for r in self.rows if (r['edge_id'], r['orientation'], r['arm'], r['condition'], r['provider']) ==
                    (edge_id, orientation, arm, condition, provider))

    def test_contradictory_transport_response_rejected(self):
        begin = start(self.select())
        event = finish(begin, status='transport_error')
        for key, value in (('rawbody', '{}'), ('http_status', 200)):
            bad = copy.deepcopy(event)
            bad[key] = value
            with self.assertRaisesRegex(ValueError, 'contradictory transport-error response'):
                self.analyze([begin, bad])

    def test_model_identity_deviations_block_attribution_without_dropping(self):
        begin = start(self.select())
        requested = begin['payload']['model']
        for model, expected in ((requested, True), ('wrong-model', False), (None, False), ('unapproved-alias', False)):
            event = finish(begin, response_model=model)
            metrics, cells = self.analyze([begin, event])
            self.assertEqual(metrics['within_model_attribution_valid'], expected)
            self.assertEqual(len(metrics['model_identity_deviations']), 0 if expected else 1)
            self.assertEqual(metrics['overall']['scheduled'], 72)
            self.assertEqual(metrics['overall']['valid_returned'], 1)
            self.assertEqual(metrics['accounting']['known_usage_finishes'], 1)

    def test_wrong_model_nonvalid_responses_also_block_attribution(self):
        begin = start(self.select())
        for status in ('invalid_answer', 'incomplete_answer', 'http_error'):
            metrics, _ = self.analyze([begin, finish(begin, status=status, response_model='wrong-model')])
            self.assertFalse(metrics['within_model_attribution_valid'])
            self.assertEqual(metrics['overall']['scheduled'], 72)
            self.assertEqual(metrics['overall']['valid_returned'], 0)
            self.assertEqual(metrics['accounting']['known_usage_finishes'], 1)
        metrics, _ = self.analyze([begin, finish(begin, status='transport_error')])
        self.assertTrue(metrics['within_model_attribution_valid'])
        self.assertEqual(metrics['accounting']['unknown_or_pending_attempts'], 1)

    def test_no_responses_preserves_every_scheduled_denominator(self):
        metrics, cells = self.analyze()
        self.assertEqual(len(cells), 72)
        self.assertEqual(len(metrics['primary_cells']), 12)
        self.assertEqual(len(metrics['model_input_rows']), 4)
        self.assertEqual(metrics['overall']['scheduled'], 72)
        self.assertEqual(metrics['overall']['unattempted'], 72)
        self.assertEqual(metrics['overall']['agreement'], 0)
        self.assertIsNone(metrics['overall']['returned_valid_agreement'])
        self.assertFalse(metrics['formal_journal_complete'])
        for row in metrics['primary_cells']:
            self.assertEqual((row['scheduled'], row['edge_presentations']), (6, 3))
        cell = next(c for c in cells if c['logical_id'] == self.select(orientation='reverse')['logical_id'])
        self.assertEqual((cell['A_id'], cell['B_id'], cell['reference']), ('n2', 'n1', 'B_SUB_A'))

    def test_missing_invalid_pending_and_valid_are_separate(self):
        selected = [self.select('e1'), self.select('e1', 'reverse'),
                    self.select('e2'), self.select('e2', 'reverse'), self.select('e3')]
        events = []
        for i, row in enumerate(selected, 1):
            begin = start(row, i)
            events.append(begin)
            if i < 5:
                events.append(finish(begin, label='A_SUB_B' if i == 1 else 'OTHER',
                                     status={1: 'ok', 2: 'invalid_answer', 3: 'ok', 4: 'transport_error'}[i]))
        metrics, _ = self.analyze(events)
        r = result_for(metrics)
        expected = {'scheduled': 6, 'attempted': 5, 'finished': 4, 'returned': 3,
                    'valid_returned': 2, 'invalid_returned': 1, 'pending': 1,
                    'unattempted': 1, 'matches': 1, 'nonmatches': 5, 'no_valid_label': 4}
        for key, value in expected.items():
            self.assertEqual(r[key], value, key)
        self.assertEqual(r['agreement'], 1 / 6)
        self.assertEqual(r['returned_valid_agreement'], 1 / 2)
        self.assertEqual(r['label_counts'], dict(A_SUB_B=1, B_SUB_A=0, EQUIVALENT=0, OTHER=1, UNRESOLVED=0))

    def test_correctness_is_not_reversal_consistency(self):
        events = []
        labels = {'e1': ('A_SUB_B', 'B_SUB_A'), 'e2': ('B_SUB_A', 'A_SUB_B'), 'e3': ('OTHER', 'OTHER')}
        for eid, pair in labels.items():
            for o, label in zip(a.ORIENTATIONS, pair):
                begin = start(self.select(eid, o), len(events) + 1)
                events.extend((begin, finish(begin, label)))
        r = result_for(self.analyze(events)[0])
        self.assertEqual(r['matches'], 2)
        self.assertEqual(r['both_orientations_correct'], 1)
        self.assertEqual(r['both_orientations_correct_rate'], 1 / 3)
        self.assertEqual(r['reversal_consistent'], 3)
        self.assertEqual(r['reversal_consistency'], 1)
        for label in ('EQUIVALENT', 'OTHER', 'UNRESOLVED'):
            with self.subTest(label=label):
                events = []
                for o in a.ORIENTATIONS:
                    begin = start(self.select(orientation=o), len(events) + 1)
                    events.extend((begin, finish(begin, label)))
                r = result_for(self.analyze(events)[0])
                self.assertEqual((r['both_orientations_correct'], r['reversal_consistent']), (0, 1))
                self.assertEqual(r['reversal_consistency'], 1 / 3)
                self.assertEqual(r['reversal_consistency_both_valid'], 1)

    def test_constant_and_containment_baselines_count_abstentions(self):
        metrics, _ = self.analyze()
        self.assertEqual(len(metrics['baselines']), 5)
        for baseline in metrics['baselines']:
            self.assertEqual(baseline['scheduled'], 6)
            if baseline['baseline'] == 'constant_A':
                self.assertEqual((baseline['matches'], baseline['agreement'], baseline['reversal_consistent']), (3, 0.5, 0))
            elif baseline['baseline'] == 'name_containment':
                self.assertEqual((baseline['matches'], baseline['abstentions']), (2, 4))
                self.assertEqual(baseline['agreement'], 1 / 3)
            else:
                self.assertEqual((baseline['matches'], baseline['abstentions']), (4, 2))
                self.assertEqual(baseline['agreement'], 2 / 3)
        for use_defs in (False, True):
            material = {'A': {'label': 'same', 'definitions': ['same']},
                        'B': {'label': 'same', 'definitions': ['same']}}
            self.assertEqual(plan.lexical_label(material, use_defs), 'UNRESOLVED')

    def test_components_overlapping_families_cues_and_contrasts(self):
        events = []
        # F succeeds on e1/e2, L only e3, D only e1: F-L=1/3, F-D=1/3.
        successes = {'F': {'e1', 'e2'}, 'L': {'e3'}, 'D': {'e1'}}
        for i, row in enumerate(self.rows, 1):
            begin = start(row, i)
            label = row['reference'] if row['edge_id'] in successes[row['arm']] else 'OTHER'
            events.extend((begin, finish(begin, label)))
        metrics, _ = self.analyze(events)
        self.assertTrue(metrics['formal_journal_complete'])
        self.assertTrue(metrics['formal_all_valid'])
        self.assertEqual([(g['node_count'], g['edge_count']) for g in metrics['components']], [(3, 2), (2, 1)])
        self.assertEqual(a.graph_components(self.edges), a.graph_components(list(reversed(self.edges))))
        for row in metrics['model_input_rows']:
            self.assertEqual([row['arms'][arm]['matches'] for arm in a.ARMS], [2, 4, 2])
            self.assertAlmostEqual(row['F_minus_L_pp'], 100 / 3)
            self.assertAlmostEqual(row['F_minus_D_pp'], 100 / 3)
        for row in metrics['leave_one_component_out_effect_ranges']:
            self.assertEqual(row['F_minus_L_pp'], {'min': -100, 'max': 100, 'defined_omissions': 2})
            self.assertEqual(row['F_minus_D_pp'], {'min': 0, 'max': 50, 'defined_omissions': 2})
        families = [g for g in metrics['strata'] if g['kind'] == 'family']
        self.assertEqual([g['edge_count'] for g in families], [2, 2])
        cue = [g for g in metrics['strata'] if g['kind'] == 'definition_cue']
        self.assertEqual([g['edge_count'] for g in cue], [2, 1])

    def test_one_component_has_undefined_leave_out_range(self):
        self.edges = self.edges[:2]
        self.rows = requests(self.edges)
        metrics, _ = self.analyze()
        for row in metrics['leave_one_component_out_effect_ranges']:
            self.assertEqual(row['F_minus_L_pp'], {'min': None, 'max': None, 'defined_omissions': 0})

    def test_smoke_exclusion_cost_liabilities_models_and_usage(self):
        rows = requests(self.edges, smoke=True)
        smoke = next(r for r in rows if r['phase'] == 'smoke' and r['provider'] == 'qwen')
        selected = [self.select(), smoke, self.select('e2'), self.select('e3')]
        events = []
        for i, row in enumerate(selected, 1):
            begin = start(row, i)
            events.append(begin)
            if i < 4:
                events.append(finish(begin, status='unknown_usage' if i == 3 else 'ok'))
        metrics, cells = self.analyze(events, rows)
        costs = metrics['accounting']
        self.assertEqual(len(cells), 72)
        self.assertEqual(metrics['overall']['matches'], 1)
        self.assertEqual(Decimal(costs['new_actual_conservative_journal_cny']), Decimal('0.000090'))
        reserve = Decimal(selected[2]['reserved_cny']) + Decimal(selected[3]['reserved_cny'])
        self.assertEqual(Decimal(costs['unknown_full_reserved_liability_cny']), reserve)
        self.assertEqual(costs['unknown_or_pending_attempts'], 2)
        self.assertEqual(costs['prior_conservative_cny'], str(plan.PAST))
        self.assertEqual(Decimal(costs['total_with_prior_cny']), plan.PAST + Decimal('0.0000900') + reserve)
        self.assertEqual(costs['validated_usage_totals'], dict(prompt_tokens=20, completion_tokens=4, total_tokens=24))
        self.assertEqual(costs['returned_model_identifiers'], {'returned-test-model': 3})
        self.assertEqual(costs['by_phase']['smoke']['known_usage_finishes'], 1)
        self.assertEqual(costs['by_phase']['formal']['known_usage_finishes'], 1)

    def test_all_failure_statuses_remain_nonmatches(self):
        for status in a.STATUSES:
            with self.subTest(status=status):
                begin = start(self.select())
                metrics, _ = self.analyze([begin, finish(begin, status=status)])
                r = result_for(metrics)
                self.assertEqual(r['status_counts'][status], 1)
                self.assertEqual(r['matches'], int(status == 'ok'))
                self.assertEqual(r['scheduled'], 6)

    def test_known_over_bound_usage_keeps_exact_liability_and_tokens(self):
        begin = start(self.select())
        # Slightly over the output bound, but the computed charge is below reserve.
        end = finish(begin, status='usage_bound_exceeded')
        costs = self.analyze([begin, end])[0]['accounting']
        self.assertEqual(costs['known_usage_finishes'], 1)
        self.assertEqual(costs['unknown_or_pending_attempts'], 0)
        self.assertEqual(Decimal(costs['new_actual_conservative_journal_cny']), Decimal(begin['reserved_cny']))
        self.assertEqual(costs['validated_usage_totals']['total_tokens'], 139)
        # A known large input bill must exceed reserve and still be retained.
        end['usage'] = {'prompt_tokens': 1000000, 'completion_tokens': 2, 'total_tokens': 1000002}
        end['response']['usage'] = end['usage']
        end['rawbody'] = json.dumps(end['response'])
        end['estimated_cost_cny'] = '2.50002'  # 1M*2.5/1M + 2*10/1M, Qwen rates.
        metrics, _ = self.analyze([begin, end])
        costs = metrics['accounting']
        self.assertEqual(result_for(metrics)['matches'], 0)
        self.assertEqual(costs['known_over_bound_finishes'], 1)
        self.assertEqual(costs['known_over_bound_conservative_journal_cny'], '2.50002')
        self.assertEqual(costs['new_actual_conservative_journal_cny'], '2.50002')
        self.assertEqual(costs['unknown_full_reserved_liability_cny'], '0')
        self.assertEqual(Decimal(costs['total_with_prior_cny']), plan.PAST + Decimal('2.50002'))
        self.assertEqual(costs['validated_usage_totals'],
                         dict(prompt_tokens=1000000, completion_tokens=2, total_tokens=1000002))
        for bad_charge in (begin['reserved_cny'], '2.50003'):
            with self.assertRaisesRegex(ValueError, 'result mismatch'):
                self.analyze([begin, {**end, 'estimated_cost_cny': bad_charge}])
        # Contradictory totals make usage unknown, even when a counter is over bound.
        end['usage']['total_tokens'] = 99
        end['status'] = 'unknown_usage'
        end['estimated_cost_cny'] = begin['reserved_cny']
        end['rawbody'] = json.dumps(end['response'])
        costs = self.analyze([begin, end])[0]['accounting']
        self.assertEqual(costs['known_usage_finishes'], 0)
        self.assertEqual(costs['unknown_or_pending_attempts'], 1)
        self.assertEqual(Decimal(costs['unknown_full_reserved_liability_cny']), Decimal(begin['reserved_cny']))

    def test_duplicate_or_tampered_journals_are_rejected(self):
        begin = start(self.select())
        end = finish(begin)
        variants = [[begin, begin], [begin, end, end], [end],
                    [begin, start(self.select('e2'))]]
        for field, value in (('request_sha256', '0' * 64), ('freeze_sha256', 'b' * 64),
                             ('logical_id', 'unknown'), ('reserved_cny', '0'),
                             ('reference', 'B_SUB_A'), ('orientation', 'reverse'),
                             ('attempt_id', 'f' * 32), ('start_utc', 'different'),
                             ('label', 'B_SUB_A'), ('label', 'OLD_LABEL'),
                             ('response_model', 'invented'), ('status', 'ignored'),
                             ('estimated_cost_cny', '0'), ('usage', {})):
            variants.append([begin, {**end, field: value}])
        variants.append([{**begin, 'payload': {}}])
        for events in variants:
            with self.subTest(events=len(events), last=events[-1].get('type')):
                with self.assertRaises(ValueError):
                    self.analyze(events)

    def test_plan_grid_and_direction_validation(self):
        a.validate_inputs(self.edges, self.rows, full_census=False)
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                a.validate_inputs(self.edges, rows, full_census=False)
        for key, value in (('reference', 'B_SUB_A'), ('orientation', 'sideways'), ('request_sha256', '0' * 64)):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                a.validate_inputs(self.edges, rows, full_census=False)
        rows = copy.deepcopy(self.rows)
        rows[0]['payload']['messages'][1]['content'] = '{}'
        rows[0]['request_sha256'] = plan.digest(plan.canonical(rows[0]['payload']).encode('utf-8'))
        with self.assertRaisesRegex(ValueError, 'binding'):
            a.validate_inputs(self.edges, rows, full_census=False)
        edges = copy.deepcopy(self.edges)
        edges[0]['source_assertion'] = 'AI_said_so'
        with self.assertRaisesRegex(ValueError, 'source assertion'):
            a.validate_inputs(edges, self.rows, full_census=False)
        with self.assertRaisesRegex(ValueError, 'census size'):
            a.validate_inputs(self.edges, self.rows)

    def test_exact_parser_and_usage_defenses(self):
        for raw in ('{"label":"A_SUB_B","label":"B_SUB_A"}', '{"n":NaN}', '{"n":Infinity}',
                    '{"n":1e9999}', '{"n":-1e9999}'):
            with self.assertRaises(ValueError):
                a.strict_json(raw)
        begin = start(self.select())
        for content in ('```json\n{"label":"A_SUB_B"}\n```', '{"label":"A_SUB_B","extra":1}',
                        '{"label":"A_SUB_B","label":"B_SUB_A"}', '{"label":null}', '[]'):
            end = finish(begin, status='invalid_answer')
            end['response']['choices'][0]['message']['content'] = content
            end['rawbody'] = json.dumps(end['response'])
            self.assertEqual(result_for(self.analyze([begin, end])[0])['valid_returned'], 0)
        for usage in ({'prompt_tokens': True, 'completion_tokens': 2},
                      {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 99},
                      {'prompt_tokens': 10, 'completion_tokens': 2, 'completion_tokens_details': {'reasoning_tokens': -1}}):
            self.assertEqual(a.usage_error(usage, begin), 'unknown_usage')
        self.assertEqual(a.usage_error({'prompt_tokens': 10, 'completion_tokens': 129}, begin), 'usage_bound_exceeded')
        self.assertEqual(a.usage_error({'prompt_tokens': 1000000, 'completion_tokens': -1}, begin), 'unknown_usage')
        self.assertEqual(a.usage_error({'prompt_tokens': 1000000, 'completion_tokens': 2, 'total_tokens': 1}, begin), 'unknown_usage')


class MainFixtureTests(unittest.TestCase):
    def test_import_is_side_effect_free(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = importlib.util.spec_from_file_location('isolated_analysis_import', a.__file__)
            module = importlib.util.module_from_spec(spec)
            with patch.object(Path, 'mkdir', side_effect=AssertionError('import created directory')), \
                 patch.object(Path, 'open', side_effect=AssertionError('import opened data/output')), \
                 patch('builtins.open', side_effect=AssertionError('import opened data/output')):
                spec.loader.exec_module(module)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_full_synthetic_census_main_exports_and_freeze_defenses(self):
        edges = []
        for i in range(93):
            child, parent = (i, i + 1) if i < 89 else (90 + 2 * (i - 89), 91 + 2 * (i - 89))
            edges.append(edge('synthetic_%03d' % i, 'node_%03d' % child, 'node_%03d' % parent,
                              cue=i < 85, families=['家族%d' % (i % 7)]))
        rows = requests(edges, smoke=True)
        a.validate_inputs(edges, rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'planning').mkdir()
            (root / 'results').mkdir()
            (root / 'data/bcio_edges.json').write_text(json.dumps(edges, ensure_ascii=False), encoding='utf-8')
            (root / 'planning/requests.provisional.json').write_text(json.dumps(rows), encoding='utf-8')
            (root / 'build_plan.py').write_bytes(Path(plan.__file__).read_bytes())
            manifest = {name: plan.digest((root / name).read_bytes()) for name in
                        ('data/bcio_edges.json', 'planning/requests.provisional.json', 'build_plan.py')}
            manifest_raw = json.dumps(manifest).encode('utf-8')
            (root / 'freeze_manifest.json').write_bytes(manifest_raw)
            # Missing logs must not manufacture real-looking analysis artifacts.
            with self.assertRaises(FileNotFoundError):
                a.main(['--root', str(root)])
            self.assertFalse((root / 'analysis').exists())
            begin = start(rows[0], binding=plan.digest(manifest_raw))
            journal = '\n'.join(json.dumps(e) for e in (begin, finish(begin))) + '\n'
            journal_path = root / 'results/events.jsonl'
            journal_path.write_text(journal[:-1], encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'torn journal'):
                a.main(['--root', str(root)])
            self.assertFalse((root / 'analysis').exists())
            journal_path.write_text(journal, encoding='utf-8')
            metrics = a.main(['--root', str(root)])
            self.assertEqual(metrics['scientific_scheduled'], 2232)
            self.assertEqual(metrics['source_edges'], 93)
            self.assertEqual(metrics['overall']['matches'], 1)
            self.assertEqual(len(metrics['primary_cells']), 12)
            self.assertEqual(len(metrics['model_input_rows']), 4)
            for row in metrics['primary_cells']:
                self.assertEqual((row['scheduled'], row['edge_presentations']), (186, 93))
            cues = [g['edge_count'] for g in metrics['strata'] if g['kind'] == 'definition_cue']
            self.assertEqual(cues, [85, 8])
            outputs = {p.name: p.read_bytes() for p in (root / 'analysis').iterdir()}
            self.assertEqual(set(outputs), {'metrics.json', 'per_cell.csv'})
            self.assertEqual(json.loads(outputs['metrics.json']), metrics)
            with (root / 'analysis/per_cell.csv').open(encoding='utf-8', newline='') as stream:
                cells = list(csv.DictReader(stream))
            self.assertEqual(len(cells), 2232)
            self.assertIn('家族', cells[0]['families'])
            a.main(['--root', str(root)])
            self.assertEqual(outputs, {p.name: p.read_bytes() for p in (root / 'analysis').iterdir()})
            (root / 'data/bcio_edges.json').write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                a.load_inputs(root)


if __name__ == '__main__':
    unittest.main()
