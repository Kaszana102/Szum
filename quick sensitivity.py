
confusion = [
    [0.7, 0.075, 0.075, 0.15],
    [0.019, 0.56, 0.17, 0.26],
    [0.076, 0.15, 0.56, 0.21],
    [0.13, 0.18, 0.17, 0.52]]
for animal_index in range(4):
    true_positive = confusion[animal_index][animal_index]
    false_positive = sum(row[animal_index] if index != animal_index else 0 for index, row in enumerate(confusion))
    true_negative = sum(
        x if row_index != animal_index and index != animal_index else 0
        for row_index, row in enumerate(confusion)
        for index, x in enumerate(row)
    )
    false_negative = sum(x if index != animal_index else 0 for index, x in enumerate(confusion[animal_index]))

    sensitivity = true_positive / (true_positive + false_negative)
    specifity = true_negative / (true_negative + false_positive)

    print(f"Class {animal_index}: sensitivity: {sensitivity:.3f}, specifity: {specifity:.3f}")
