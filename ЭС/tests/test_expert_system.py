"""Проверки цепочки вывода, исправления и граничных случаев."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert_system import ColdExpertSystem, PatientSymptoms, Decision, SCENARIOS, evaluate


@pytest.mark.parametrize('scenario,expected',[
    ('orvi',['R1','R6']),('flu_warning',['R2','R4','R6']),
    ('urgent',['R2','R5','R6']),('no_match',['R7'])])
def test_expected_reasoning_chain(scenario, expected):
    result=evaluate(SCENARIOS[scenario])
    assert [rule['id'] for rule in result['rules']] == expected


def test_fix_preserves_warning_in_final_decision():
    result=evaluate(SCENARIOS['flu_warning'])
    # Демонстрационный ошибочный вариант решения: гипотеза есть, предупреждение потеряно.
    before={'hypotheses':[item['name'] for item in result['hypotheses']], 'warnings':[]}
    assert before['warnings'] != result['decision']['warnings']
    assert 'более трёх дней' in result['decision']['warnings'][0]
    assert 'предупреждений' in result['decision']['action']


def test_urgent_warning_has_priority_without_disease_match():
    result=evaluate(PatientSymptoms(temperature=36.6,shortness_of_breath=True))
    assert [rule['id'] for rule in result['rules']] == ['R5','R6']
    assert 'срочная' in result['decision']['action']


def test_reset_removes_previous_patient():
    engine=ColdExpertSystem()
    engine.declare(SCENARIOS['urgent']);engine.run()
    engine.reset();engine.declare(SCENARIOS['no_match']);engine.run()
    assert [rule['id'] for rule in engine.triggered_rules] == ['R7']
    assert len(engine.find(Decision)) == 1


@pytest.mark.parametrize('symptoms',[PatientSymptoms(temperature=float('nan')),
    PatientSymptoms(temperature=100),PatientSymptoms(days_with_fever=-1)])
def test_invalid_input(symptoms):
    with pytest.raises(ValueError):evaluate(symptoms)
