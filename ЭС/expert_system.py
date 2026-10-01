"""Практика ЭС 3. Учебная система по теме простудных заболеваний.

Правила используют заданные симптомы. Они демонстрируют логический вывод,
а не устанавливают медицинский диагноз. Проценты достоверности не назначаю:
для них у меня нет проверенной статистики.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math


class Fact:
    """Общий тип для входных, промежуточных и выходных фактов."""


@dataclass(frozen=True)
class PatientSymptoms(Fact):
    temperature: float = 36.6
    runny_nose: bool = False
    cough: bool = False
    chills: bool = False
    weakness: bool = False
    headache: bool = False
    severe_sore_throat: bool = False
    difficult_swallowing: bool = False
    shortness_of_breath: bool = False
    days_with_fever: int = 0

    def validate(self):
        if not math.isfinite(self.temperature) or not 30 <= self.temperature <= 45:
            raise ValueError('Температура должна быть числом от 30 до 45 °C.')
        if not isinstance(self.days_with_fever, int) or self.days_with_fever < 0:
            raise ValueError('Количество дней должно быть целым неотрицательным числом.')


@dataclass(frozen=True)
class Hypothesis(Fact):
    name: str
    rule: str


@dataclass(frozen=True)
class WarningSign(Fact):
    name: str
    urgent: bool
    rule: str


@dataclass(frozen=True)
class Decision(Fact):
    hypotheses: tuple
    warnings: tuple
    action: str


def Rule(rule_id, name, phase):
    """Декоратор отмечает методы, которые движок должен выполнить как правила."""
    def decorate(method):
        method.rule_info = (rule_id, name, phase)
        return method
    return decorate


class KnowledgeEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        # Перед каждым пациентом очищаю факты и историю, чтобы случаи не смешивались.
        self.facts = []
        self.triggered_rules = []

    def declare(self, fact):
        if fact not in self.facts:
            self.facts.append(fact)

    def find(self, fact_type):
        return [fact for fact in self.facts if isinstance(fact, fact_type)]

    def run(self):
        if not self.find(PatientSymptoms):
            raise ValueError('Сначала нужно передать симптомы через declare().')
        self.find(PatientSymptoms)[0].validate()
        rules = []
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, 'rule_info'):
                rules.append(method)
        # Сначала симптомы создают промежуточные факты, затем решение читает эти факты.
        rules.sort(key=lambda method: (method.rule_info[2], method.rule_info[0]))
        fired = {item['id'] for item in self.triggered_rules}
        while True:
            changed = False
            for method in rules:
                rule_id, name, phase = method.rule_info
                if rule_id not in fired and method():
                    fired.add(rule_id)
                    self.triggered_rules.append({'id': rule_id, 'name': name})
                    changed = True
            if not changed:
                break


class ColdExpertSystem(KnowledgeEngine):
    @property
    def symptoms(self):
        return self.find(PatientSymptoms)[0]

    @Rule('R1', 'Признаки ОРВИ', 1)
    def viral_signs(self):
        s = self.symptoms
        if s.runny_nose and s.cough and 37 <= s.temperature <= 38:
            self.declare(Hypothesis('Признаки, похожие на ОРВИ', 'R1'))
            return True
        return False

    @Rule('R2', 'Признаки гриппа', 1)
    def flu_signs(self):
        s = self.symptoms
        if s.temperature >= 38.5 and s.chills and s.weakness and s.headache:
            self.declare(Hypothesis('Признаки, похожие на грипп', 'R2'))
            return True
        return False

    @Rule('R3', 'Признаки выраженного воспаления горла', 1)
    def throat_signs(self):
        s = self.symptoms
        if s.severe_sore_throat and s.difficult_swallowing and s.temperature >= 38:
            self.declare(Hypothesis('Признаки выраженного воспаления горла', 'R3'))
            return True
        return False

    @Rule('R4', 'Длительная температура и слабость', 1)
    def prolonged_fever(self):
        s = self.symptoms
        if s.temperature >= 37 and s.days_with_fever > 3 and s.weakness:
            self.declare(WarningSign('Температура держится более трёх дней при слабости', False, 'R4'))
            return True
        return False

    @Rule('R5', 'Затруднённое дыхание', 1)
    def breathing_warning(self):
        if self.symptoms.shortness_of_breath:
            self.declare(WarningSign('Затруднённое дыхание', True, 'R5'))
            return True
        return False

    @Rule('R6', 'Решение по гипотезам и предупреждениям', 2)
    def final_decision(self):
        hypotheses = self.find(Hypothesis)
        warnings = self.find(WarningSign)
        if self.find(Decision) or not (hypotheses or warnings):
            return False
        # Предупреждения проверяю отдельно: они не конкурируют с названием заболевания.
        if any(warning.urgent for warning in warnings):
            action = 'Нужна срочная медицинская оценка из-за затруднённого дыхания.'
        elif warnings:
            action = 'Нужна консультация медицинского специалиста с учётом отмеченных предупреждений.'
        else:
            action = 'Получена учебная гипотеза по правилам. Для диагноза нужна оценка медицинского специалиста.'
        self.declare(Decision(tuple(item.name for item in hypotheses), tuple(item.name for item in warnings), action))
        return True

    @Rule('R7', 'Недостаточно совпадений с правилами', 2)
    def no_match(self):
        if not self.find(Decision) and not self.find(Hypothesis) and not self.find(WarningSign):
            self.declare(Decision((), (), 'Правила не дали гипотезу. Это не означает отсутствие заболевания.'))
            return True
        return False


SCENARIOS = {
    'orvi': PatientSymptoms(37.6, runny_nose=True, cough=True, weakness=True, days_with_fever=2),
    'flu_warning': PatientSymptoms(39.1, cough=True, chills=True, weakness=True, headache=True, days_with_fever=4),
    'urgent': PatientSymptoms(39.1, cough=True, chills=True, weakness=True, headache=True, shortness_of_breath=True, days_with_fever=2),
    'no_match': PatientSymptoms(),
}


def evaluate(symptoms):
    engine = ColdExpertSystem()
    engine.declare(symptoms)
    engine.run()
    return {'input': asdict(symptoms), 'rules': engine.triggered_rules,
            'hypotheses': [asdict(item) for item in engine.find(Hypothesis)],
            'warnings': [asdict(item) for item in engine.find(WarningSign)],
            'decision': asdict(engine.find(Decision)[0])}


def format_result(result):
    lines = ['УЧЕБНАЯ ЭКСПЕРТНАЯ СИСТЕМА', 'Тема: простудные заболевания', 'Входные факты:']
    lines += [f'  {key}: {value}' for key, value in result['input'].items()]
    lines += ['', 'Сработавшие правила:']
    lines += [f'  {rule["id"]}: {rule["name"]}' for rule in result['rules']]
    decision = result['decision']
    lines += ['', 'Учебные гипотезы:'] + ['  '+name for name in decision['hypotheses']]
    if not decision['hypotheses']:
        lines.append('  Нет совпадений с правилами гипотез.')
    lines += ['', 'Предупреждения:'] + ['  '+name for name in decision['warnings']]
    if not decision['warnings']:
        lines.append('  Не отмечены правилами. Это не медицинское заключение.')
    lines += ['', 'Итог:', decision['action']]
    return '\n'.join(lines)


def ask_bool(question):
    while True:
        answer = input(question+' (да/нет): ').strip().lower()
        if answer in {'да', 'д', 'нет', 'н'}:
            return answer in {'да', 'д'}
        print('Введите да или нет.')


def interactive_input():
    while True:
        try:
            temperature = float(input('Температура, °C: ').replace(',', '.'))
            days = int(input('Количество дней с температурой: '))
            PatientSymptoms(temperature=temperature, days_with_fever=days).validate()
            break
        except ValueError as error:
            print('Проверьте ввод:', error)
    questions = {'runny_nose': 'Есть насморк', 'cough': 'Есть кашель', 'chills': 'Есть озноб',
                 'weakness': 'Есть слабость', 'headache': 'Есть головная боль',
                 'severe_sore_throat': 'Есть сильная боль в горле',
                 'difficult_swallowing': 'Трудно глотать', 'shortness_of_breath': 'Есть затруднённое дыхание'}
    flags = {key: ask_bool(question) for key, question in questions.items()}
    return PatientSymptoms(temperature=temperature, days_with_fever=days, **flags)


def main():
    parser = argparse.ArgumentParser(description='Учебная экспертная система')
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--all', action='store_true', help='Показать и сохранить все демонстрационные сценарии')
    args = parser.parse_args()
    if args.all:
        results = {name: evaluate(symptoms) for name, symptoms in SCENARIOS.items()}
        folder = Path(__file__).resolve().parent / 'results'
        folder.mkdir(exist_ok=True)
        (folder/'scenarios.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        for name, result in results.items():
            text = format_result(result)
            (folder/(name+'.txt')).write_text(text, encoding='utf-8')
            print('\nСценарий:', name, '\n'+text)
    else:
        symptoms = SCENARIOS[args.scenario] if args.scenario else interactive_input()
        print(format_result(evaluate(symptoms)))


if __name__ == '__main__':
    main()
