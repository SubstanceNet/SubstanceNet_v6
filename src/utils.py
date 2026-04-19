import torch
import numpy as np
import math
from sklearn.metrics import mutual_info_score

def compute_integration_info(wave_funcs):
    """
    Integrated Information Theory (IIT) metric: mutual information between
    phase dimensions as a proxy for information integration.

    ⚠️ UNUSED IN v6: retained from v5 wave-formalism era as reference
    implementation. Not invoked by any publication experiment (exp01-06).
    Candidate for v7 activation if IIT-based consciousness metrics are
    reintroduced.
    """
    if wave_funcs is None or wave_funcs.numel() == 0:
        return 0.0
    if wave_funcs.is_cuda:
        wave_funcs = wave_funcs.cpu()
    _, phase = wave_funcs.chunk(2, dim=-1)
    phase_np = phase[0].detach().numpy()
    if phase_np.size == 0 or phase_np.shape[1] < 2:
        return 0.0
    if np.any(np.isnan(phase_np)) or np.any(np.isinf(phase_np)):
        return 0.0
    if np.allclose(phase_np[:, 0].min(), phase_np[:, 0].max()) or \
       np.allclose(phase_np[:, 1].min(), phase_np[:, 1].max()):
        return 0.0
    bins1 = np.linspace(phase_np[:, 0].min(), phase_np[:, 0].max(), 10 + 1)
    bins2 = np.linspace(phase_np[:, 1].min(), phase_np[:, 1].max(), 10 + 1)
    disc1 = np.digitize(phase_np[:, 0], bins=bins1[:-1], right=False)
    disc2 = np.digitize(phase_np[:, 1], bins=bins2[:-1], right=False)
    try:
        return mutual_info_score(disc1, disc2)
    except ValueError:
        return 0.0

def compute_phi_approx(wave_funcs):
    """
    Approximation of IIT's Φ (integrated information) via minimum
    information partition over phase dimensions.

    ⚠️ UNUSED IN v6: retained from v5 wave-formalism era as reference
    implementation. Not invoked by any publication experiment (exp01-06).
    Candidate for v7 activation if IIT-based consciousness metrics are
    reintroduced.
    """
    if wave_funcs is None or wave_funcs.numel() == 0:
        return 0.0
    if wave_funcs.is_cuda:
        wave_funcs = wave_funcs.cpu()
    _, phase = wave_funcs.chunk(2, dim=-1)
    phase_np = phase[0].detach().numpy()
    if phase_np.size == 0 or phase_np.shape[1] == 0:
        return 0.0
    if np.any(np.isnan(phase_np)) or np.any(np.isinf(phase_np)):
        return 0.0
    N = phase_np.shape[1]
    if N < 4:
        return 0.0
    phi = 0.0
    min_mi = float('inf')
    possible_splits = [N // 4, N // 2, (3 * N) // 4]
    sample_splits = [s for s in possible_splits if 0 < s < N]
    if not sample_splits:
        return 0.0
    for i in sample_splits:
        part1 = phase_np[:, :i]
        part2 = phase_np[:, i:]
        if part1.size == 0 or part2.size == 0:
            continue
        # Use more robust binning
        if np.ptp(part1) == 0:
            bins1 = [part1.min(), part1.max() + 1e-6]
        else:
            bins1 = np.linspace(part1.min(), part1.max(), 10)
        if np.ptp(part2) == 0:
            bins2 = [part2.min(), part2.max() + 1e-6]
        else:
            bins2 = np.linspace(part2.min(), part2.max(), 10)
        
        disc1 = np.digitize(part1.flatten(), bins=bins1)
        disc2 = np.digitize(part2.flatten(), bins=bins2)
        if len(disc1) == 0 or len(disc2) == 0:
            continue
        try:
            mi = mutual_info_score(disc1, disc2)
            min_mi = min(min_mi, mi)
            phi += mi
        except ValueError:
            pass
    if not sample_splits or min_mi == float('inf'):
        return 0.0
    phi = phi / len(sample_splits) - min_mi
    return phi / N if N > 0 else 0.0

def generate_synthetic_cognitive_data(batch_size, seq_len, num_modules, task_type="raven", num_classes=2, abstract_classes=3, print_sample=False):
    if task_type == "raven":
            data = torch.zeros(batch_size, 3, 3, dtype=torch.float64)
            labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
            abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
            for i in range(batch_size):
                base_pattern = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float64)
                variation = torch.randn(3, 3, dtype=torch.float64) * 0.2
                pattern = base_pattern + variation
                if labels[i] == 1:
                    pattern[2, 2] = 1 + variation[2, 2]
                else:
                    pattern[2, 2] = 0 + variation[2, 2]
                data[i] = pattern
                abstract_labels[i] = 0 if labels[i] == 0 else 2
                
            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_pattern = data[0].tolist()
                print(f"Raven - 3x3 сетка с паттернами {sample_pattern} с вариацией ±0.2 - Задача предсказать значение 9-й ячейки (0 или 1).")
                
            return data, labels, abstract_labels

    elif task_type == "logic":
            data = torch.zeros(batch_size, 2, num_modules, dtype=torch.float64)
            labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
            abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
            for i in range(batch_size):
                A = torch.tensor([1, 0, 1], dtype=torch.float64) if labels[i] == 1 else torch.tensor([0, 1, 0], dtype=torch.float64)
                B = torch.tensor([0, 0, 1], dtype=torch.float64)
                data[i, 0] = A
                data[i, 1] = B
                abstract_labels[i] = 1
                
            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_A = data[0, 0].tolist()
                sample_B = data[0, 1].tolist()
                print(f"Logic - Два паттерна {sample_A} и {sample_B} - Задача определить истинность первого паттерна (0 или 1).")
                
            return data, labels, abstract_labels

    elif task_type == "memory":
        data = torch.zeros(batch_size, seq_len, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        for i in range(batch_size):
            sequence = torch.randint(0, 2, (seq_len, num_modules), dtype=torch.float64)
            data[i] = sequence + torch.randn(seq_len, num_modules, dtype=torch.float64) * 0.1
            labels[i] = int(sequence[-1, 0])
            abstract_labels[i] = 2

            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_sequence = data[0].tolist()
                print(f"Memory - Последовательность из {seq_len} паттернов {sample_sequence} с шумом ±0.1 - Задача предсказать последний элемент последовательности (0 или 1).")

        return data, labels, abstract_labels

    elif task_type == "analogy":
        data = torch.zeros(batch_size, 3, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        for i in range(batch_size):
            A = torch.tensor([1, 0, 1], dtype=torch.float64)
            B = torch.tensor([1, 0, 1], dtype=torch.float64) if labels[i] == 1 else torch.tensor([0, 1, 0], dtype=torch.float64)
            C = torch.tensor([0, 1, 0], dtype=torch.float64)
            data[i, 0] = A
            data[i, 1] = B
            data[i, 2] = C
            abstract_labels[i] = 1 if labels[i] == 1 else 0

            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_A = data[0, 0].tolist()
                sample_B = data[0, 1].tolist()
                sample_C = data[0, 2].tolist()
                print(f"Analogy - Три паттерна {sample_A}, {sample_B}, {sample_C} - Задача предсказать четвёртый паттерн (0 или 1).")
        
        return data, labels, abstract_labels

    elif task_type == "categorization":
        data = torch.zeros(batch_size, seq_len, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        for i in range(batch_size):
            for j in range(seq_len):
                if labels[i] == 0:
                    data[i, j] = torch.tensor([1, 0, 1], dtype=torch.float64)
                else:
                    data[i, j] = torch.tensor([0, 1, 0], dtype=torch.float64)
                data[i, j] += torch.randn(num_modules, dtype=torch.float64) * 0.1
            abstract_labels[i] = 1
           
            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_patterns = data[0].tolist()
                print(f"Categorization - Последовательность из {seq_len} паттернов {sample_patterns} с шумом ±0.1 - Задача классифицировать последовательность (0 или 1).")
        
        return data, labels, abstract_labels

    elif task_type == "insight":
        # Исправленная задача insight: найти скрытую симметрию
        data = torch.zeros(batch_size, seq_len, num_modules, dtype=torch.float64)
        labels = torch.zeros(batch_size, dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        
        for i in range(batch_size):
            # Случайно выбираем тип паттерна
            pattern_type = torch.rand(1).item() > 0.5
            
            for j in range(seq_len):
                if pattern_type:  # Симметричный паттерн (метка = 1)
                    # Создаем симметричную структуру относительно центра
                    center = seq_len // 2
                    dist_from_center = abs(j - center)
                    x = 1.0 - dist_from_center * 0.3 + torch.randn(1).item() * 0.1
                    y = 0.5 + dist_from_center * 0.2 + torch.randn(1).item() * 0.1
                    z = x + y  # Связанная компонента
                else:  # Асимметричный паттерн (метка = 0)
                    # Создаем случайную структуру
                    x = j * 0.3 + torch.randn(1).item() * 0.2
                    y = (seq_len - j) * 0.2 + torch.randn(1).item() * 0.2
                    z = torch.randn(1).item() * 0.5  # Несвязанная компонента
                
                data[i, j] = torch.tensor([x, y, z], dtype=torch.float64)
            
            labels[i] = 1 if pattern_type else 0
            abstract_labels[i] = 2
           
        if print_sample and batch_size > 0:
            print(f"Insight - Найти скрытую симметрию в последовательности")
        
        return data, labels, abstract_labels

    elif task_type == "spatial":
        data = torch.zeros(batch_size, 2, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        for i in range(batch_size):
            figure1 = torch.tensor([1, 0, 1], dtype=torch.float64)
            if labels[i] == 1:
                figure2 = torch.tensor([0, 1, 0], dtype=torch.float64)
            else:
                figure2 = figure1 + torch.randn(num_modules, dtype=torch.float64) * 0.05
            data[i, 0] = figure1
            data[i, 1] = figure2
            abstract_labels[i] = 1
           
            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_figure1 = data[0, 0].tolist()
                sample_figure2 = data[0, 1].tolist()
                print(f"Spatial - Два паттерна {sample_figure1} и {sample_figure2} с шумом ±0.05 - Задача определить, являются ли они повёрнутыми версиями друг друга (0 или 1).")
        
        return data, labels, abstract_labels

    elif task_type == "numerical":
        data = torch.zeros(batch_size, 4, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        for i in range(batch_size):
            patterns = [
                torch.tensor([1, 0, 1], dtype=torch.float64),
                torch.tensor([0, 1, 0], dtype=torch.float64),
                torch.tensor([1, 1, 0], dtype=torch.float64),
                torch.tensor([0, 1, 1], dtype=torch.float64)
            ]
            next_pattern = torch.tensor([1, 0, 1], dtype=torch.float64) if labels[i] == 1 else torch.tensor([0, 0, 1], dtype=torch.float64)
            for j in range(4):
                data[i, j] = patterns[j]
            data[i, 0, :] = next_pattern
            abstract_labels[i] = 1
           
            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_patterns = []
                for j in range(4):
                    sample_patterns.append(data[0, j].tolist())
                print(f"Numerical - Четыре паттерна {sample_patterns} - Задача предсказать пятый паттерн (0 или 1).")
        
        return data, labels, abstract_labels

    elif task_type == "verbal":
        data = torch.zeros(batch_size, 2, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        for i in range(batch_size):
            word1 = torch.tensor([1, 0, 1], dtype=torch.float64)
            word2 = torch.tensor([1, 0, 1], dtype=torch.float64) if labels[i] == 1 else torch.tensor([0, 1, 0], dtype=torch.float64)
            data[i, 0] = word1
            data[i, 1] = word2
            abstract_labels[i] = 1
            
            # Вывод условия задачи, если требуется
            if print_sample and batch_size > 0:
                sample_word1 = data[0, 0].tolist()
                sample_word2 = data[0, 1].tolist()
                print(f"Verbal - Два паттерна {sample_word1} и {sample_word2} - Задача определить, являются ли они синонимами (0 или 1).")
        
        return data, labels, abstract_labels

    elif task_type == "emotional":
        # Исправленная задача emotional: распознавание эмоционального паттерна
        data = torch.zeros(batch_size, 2, num_modules, dtype=torch.float64)
        labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)
        abstract_labels = torch.randint(0, abstract_classes, (batch_size,), dtype=torch.long)
        
        for i in range(batch_size):
            if labels[i] == 1:  # Позитивный паттерн
                # Восходящий тренд + высокая энергия
                description = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
                emotion = torch.tensor([0.7, 0.8, 0.9], dtype=torch.float64)
            else:  # Негативный паттерн
                # Нисходящий тренд + низкая энергия
                description = torch.tensor([0.8, 0.5, 0.2], dtype=torch.float64)
                emotion = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float64)
            
            # Добавляем небольшой шум
            description += torch.randn(num_modules, dtype=torch.float64) * 0.1
            emotion += torch.randn(num_modules, dtype=torch.float64) * 0.1
            
            data[i, 0] = description
            data[i, 1] = emotion
            abstract_labels[i] = 1
            
        if print_sample and batch_size > 0:
            print(f"Emotional - Распознать эмоциональный паттерн (позитивный=1, негативный=0)")
        
        return data, labels, abstract_labels
