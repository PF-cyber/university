class CocomoCalculator:
    # Коэффициенты для разных типов проектов: [a, b, c, d]
    coefficients = {
        "organic": [3.2, 1.05, 2.5, 0.38],
        "semi-detached": [3.0, 1.12, 2.5, 0.35],
        "embedded": [2.8, 1.20, 2.5, 0.32]
    }

    # Словарь стоимостных драйверов (упрощенный, только несколько примеров)
    # Каждый драйвер имеет множитель для каждого уровня: VL, L, N, H, VH, XH
    cost_drivers = {
        "RELY": {"VL": 0.75, "L": 0.88, "N": 1.00, "H": 1.15, "VH": 1.40, "XH": None}, # Надежность
        "PEXP": {"VL": 1.29, "L": 1.13, "N": 1.00, "H": 0.91, "VH": 0.82, "XH": None}, # Опыт персонала
        "CPLX": {"VL": 0.70, "L": 0.85, "N": 1.00, "H": 1.17, "VH": 1.34, "XH": 1.74}, # Сложность
        "LTEX": {"VL": 1.21, "L": 1.10, "N": 1.00, "H": 0.91, "VH": 0.84, "XH": None}, # Опыт в технологии
        "SCED": {"VL": None, "L": 1.10, "N": 1.00, "H": 1.08, "VH": 1.23, "XH": None}  # Сжатие сроков
    }

    def __init__(self, project_type, kloc, cost_driver_values):
        """
        Инициализация калькулятора.
        :param project_type: Тип проекта ('organic', 'semi-detached', 'embedded')
        :param kloc: Размер проекта в тысячах строк кода (KSLOC)
        :param cost_driver_values: Словарь с рейтингами для драйверов.
                                   Например: {'RELY': 'H', 'PEXP': 'N', 'CPLX': 'H'}
        """
        self.project_type = project_type
        self.kloc = kloc
        self.cost_driver_values = cost_driver_values
        self.a, self.b, self.c, self.d = self.coefficients[project_type]
        self.eaf = self.calculate_eaf()

    def calculate_eaf(self):
        """Рассчитывает поправочный коэффициент EAF."""
        eaf = 1.0
        for driver, rating in self.cost_driver_values.items():
            if driver in self.cost_drivers and rating in self.cost_drivers[driver]:
                multiplier = self.cost_drivers[driver][rating]
                if multiplier is not None:
                    eaf *= multiplier
                else:
                    print(f"Предупреждение: Для драйвера {driver} рейтинг {rating} не поддерживается или не имеет множителя. Принят за 1.0.")
            else:
                print(f"Предупреждение: Драйвер {driver} или рейтинг {rating} не найдены. Приняты за 1.0.")
        return eaf

    def calculate(self):
        """Выполняет все расчеты COCOMO."""
        # Расчет трудозатрат (PM)
        pm_nominal = self.a * (self.kloc ** self.b) # Без поправок
        pm_adjusted = pm_nominal * self.eaf          # С поправкой на EAF

        # Расчет времени разработки (TDEV)
        tdev_nominal = self.c * (pm_nominal ** self.d)
        tdev_adjusted = self.c * (pm_adjusted ** self.d)

        # Расчет средней численности команды
        avg_team_size = pm_adjusted / tdev_adjusted

        results = {
            "PM (номинально)": round(pm_nominal, 2),
            "EAF": round(self.eaf, 2),
            "PM (скорректировано)": round(pm_adjusted, 2),
            "TDEV (номинально, мес.)": round(tdev_nominal, 2),
            "TDEV (скорректировано, мес.)": round(tdev_adjusted, 2),
            "Средний размер команды (чел.)": round(avg_team_size, 2)
        }
        return results

    def print_results(self, results):
        """Красиво выводит результаты."""
        print("\n--- РЕЗУЛЬТАТЫ ОЦЕНКИ ПО COCOMO ---")
        print(f"Тип проекта: {self.project_type.capitalize()}")
        print(f"Размер (KSLOC): {self.kloc}")
        print(f"Поправочный коэффициент (EAF): {results['EAF']}")
        print("---")
        print(f"Трудозатраты (номинальные): {results['PM (номинально)']} человеко-месяцев")
        print(f"Трудозатраты (скорректированные): {results['PM (скорректировано)']} человеко-месяцев")
        print(f"Время разработки (номинальное): {results['TDEV (номинально, мес.)']} месяцев")
        print(f"Время разработки (скорректированное): {results['TDEV (скорректировано, мес.)']} месяцев")
        print(f"Средний размер команды: {results['Средний размер команды (чел.)']} человек")
        print("------------------------------------")

# Пример использования для проекта мессенджера "ChatCorp"
if __name__ == "__main__":
    # Параметры проекта
    project_type = "semi-detached"
    size_kloc = 15
    # Значения выбранных стоимостных драйверов
    drivers = {
        "RELY": "H",    # Высокая надежность
        "PEXP": "N",    # Нормальный опыт
        "CPLX": "H",    # Высокая сложность
        "LTEX": "L",    # Низкий опыт в технологии (примем за Low, хотя выше был пример с Normal)
        "SCED": "H"     # Сжатые сроки
    }

    # Создаем калькулятор и производим расчет
    calculator = CocomoCalculator(project_type, size_kloc, drivers)
    results = calculator.calculate()
    calculator.print_results(results)